"""
daytona_runner.py — Orchestrator + Worker architecture

Orchestrator (this process):
  Coordinates the 4-phase pipeline, streams SSE events to the client.

Workers (Daytona sandboxes):
  - Orchestrator sandbox  : recon → planning → report  (sequential, 1 sandbox)
  - Suite worker sandboxes: one per suite, bootstrapped in parallel WHILE
    recon + planning run so the bootstrap cost is hidden from the user.

Pipeline flow:
  1. Provision orchestrator sandbox + pre-spin N worker sandboxes in parallel
  2. Orchestrator runs recon + planning  (worker sandboxes bootstrap concurrently)
  3. Workers are ready by the time planning finishes → dispatch suites immediately
  4. Collect suite results as they arrive (streaming)
  5. Orchestrator runs report
  6. All sandboxes torn down in finally block
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import AsyncIterator

from daytona_sdk import AsyncDaytona, DaytonaConfig, CreateSandboxFromSnapshotParams

from app.models.schemas import FinalReport, RedTeamRequest

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # .../fairsight-backend/
_WORKSPACE = "/tmp/fairsight"
_SENTINEL = "FAIRSIGHT_RESULT:"

# Only httpx is needed — llm_client.py calls all providers via REST (no SDK).
# anthropic package is NOT required: Anthropic API is called directly via httpx.
_SANDBOX_DEPS = [
    "tavily-python==0.3.3",
    "httpx==0.28.1",
    "pydantic>=2.9.0",
    "python-dotenv==1.0.1",
]


# ─── Daytona client ───────────────────────────────────────────────────────────

def _make_client() -> AsyncDaytona:
    return AsyncDaytona(
        DaytonaConfig(
            api_key=os.environ["DAYTONA_API_KEY"],
            server_url=os.environ.get(
                "DAYTONA_SERVER_URL", "https://app.daytona.io/api"
            ),
        )
    )


# ─── Async helpers ────────────────────────────────────────────────────────────

async def _aexec(sandbox, cmd: str) -> str:
    result = await sandbox.process.exec(cmd)
    return getattr(result, "output", "") or ""


async def _with_heartbeat(coro, phase: str, interval: float = 10.0):
    """
    Run an awaitable while yielding SSE heartbeat ticks every `interval`
    seconds so the client knows the phase is still running.
    Yields heartbeat dicts; the final value is the coroutine result.
    """
    task = asyncio.ensure_future(coro)
    elapsed = 0.0
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            elapsed += interval
            yield {"phase": phase, "status": "running", "message": f"Still working… ({int(elapsed)}s elapsed)"}
        except Exception:
            break
    # Propagate any exception from the task
    yield task.result()  # raises if task raised


async def _acode_run(sandbox, code: str, poll_interval: float = 3.0, max_wait: int = 360) -> str:
    """Run Python code in the sandbox via a background process + polling.

    Avoids Daytona's per-exec timeout entirely: the script runs detached
    (nohup ... &) so the exec call that launches it returns immediately.
    We then poll for a 'done' marker file every poll_interval seconds.
    Each poll exec is short-lived and cannot timeout.

    max_wait: total seconds to wait before giving up (default 6 min).
    """
    uid = uuid.uuid4().hex
    tmp = f"/tmp/_fs_{uid}.py"
    out = f"/tmp/_fs_out_{uid}.txt"
    done = f"/tmp/_fs_done_{uid}"

    encoded = base64.b64encode(code.encode()).decode()

    # Step 1: write the code file
    write_cmd = (
        f"python3 -c \"import base64; "
        f"open('{tmp}','w').write(base64.b64decode('{encoded}').decode())\""
    )
    await sandbox.process.exec(write_cmd, timeout=30)

    # Step 2: launch in background — exec returns immediately
    bg_cmd = (
        f"nohup sh -c "
        f"'env PYTHONPATH={_WORKSPACE} python3 {tmp} > {out} 2>&1; touch {done}' "
        f"> /dev/null 2>&1 &"
    )
    await sandbox.process.exec(bg_cmd, timeout=10)

    # Step 3: poll until done marker appears
    polls = int(max_wait / poll_interval)
    for _ in range(polls):
        await asyncio.sleep(poll_interval)
        check = await sandbox.process.exec(
            f"test -f {done} && echo done || echo waiting", timeout=10
        )
        if (getattr(check, "output", "") or "").strip() == "done":
            break
    else:
        logger.warning("_acode_run: timed out after %ds waiting for %s", max_wait, done)

    # Step 4: read output file
    read_result = await sandbox.process.exec(f"cat {out} 2>/dev/null || echo ''", timeout=15)
    return (getattr(read_result, "output", "") or getattr(read_result, "result", "") or "").strip()


# ─── Sandbox bootstrap ────────────────────────────────────────────────────────

async def _bootstrap(sandbox, install_deps: bool = True):
    """
    Upload all project Python source files, and optionally install deps.
    Yields progress dicts that the caller can forward as SSE events.

    Files are written SEQUENTIALLY — one exec call per file, keeping each
    command small (<15 KB). Parallel or large-batch approaches hit Daytona
    exec command-length/concurrency limits and silently drop writes.

    install_deps=False for worker sandboxes — they share the same Python
    environment snapshot so pip install is not needed, saving 30-60s per worker.
    """
    py_files = [
        p for p in _PROJECT_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]
    total = len(py_files)

    yield {"phase": "sandbox", "status": "uploading", "message": f"Uploading {total} source files…", "done": 0, "total": total}

    await _aexec(sandbox, f"mkdir -p {_WORKSPACE}")

    dirs_created: set[str] = set()
    for i, p in enumerate(py_files, 1):
        dest = f"{_WORKSPACE}/{p.relative_to(_PROJECT_ROOT)}"
        dir_path = dest.rsplit("/", 1)[0]
        if dir_path not in dirs_created:
            await _aexec(sandbox, f"mkdir -p {dir_path}")
            dirs_created.add(dir_path)
        b64 = base64.b64encode(p.read_bytes()).decode()
        out = await _aexec(
            sandbox,
            f"python3 -c \"import base64; "
            f"open('{dest}','wb').write(base64.b64decode('{b64}'))\""
        )
        if out.strip():
            logger.warning("Write %s stderr: %s", dest, out.strip())
        # Emit progress every 5 files (and on the last one)
        if i % 5 == 0 or i == total:
            yield {"phase": "sandbox", "status": "uploading", "message": f"Uploaded {i}/{total} files", "done": i, "total": total}

    logger.info("Wrote %d source files to sandbox", total)

    if install_deps:
        yield {"phase": "sandbox", "status": "installing", "message": "Installing dependencies…"}
        deps_str = " ".join(_SANDBOX_DEPS)
        pip_out = await _aexec(sandbox, f"python3 -m pip install -q {deps_str} 2>&1")
        logger.info("pip install: %s", pip_out.strip()[-300:] if pip_out.strip() else "ok")

    check = await _aexec(sandbox, f"ls {_WORKSPACE}/app/ 2>&1 | head -10")
    logger.info("Sandbox bootstrapped (deps=%s): %d files | app/ contents: %s", install_deps, total, check.strip())


# ─── Code template helpers ────────────────────────────────────────────────────

def _env_setup(req: RedTeamRequest) -> str:
    """
    Python snippet injected at the top of every phase code block.
    Sets env vars and sys.path so imports work inside the sandbox.

    FairSight's internal agents read FAIRSIGHT_API_KEY / FAIRSIGHT_MODEL /
    FAIRSIGHT_ENDPOINT — sourced entirely from the request, not server env.
    TAVILY_API_KEY is the only server-side key (recon web search).
    """
    # PYTHONPATH=/workspace is set at the shell level in _acode_run, so
    # `import app.*` works without any sys.path manipulation here.
    return "\n".join([
        "import os, sys, json, asyncio",
        f"os.environ['FAIRSIGHT_API_KEY']  = {repr(req.fairsight_api_key)}",
        f"os.environ['FAIRSIGHT_MODEL']    = {repr(req.fairsight_model)}",
        f"os.environ['FAIRSIGHT_ENDPOINT'] = {repr(req.fairsight_endpoint or '')}",
        f"os.environ['TAVILY_API_KEY']     = {repr(os.environ['TAVILY_API_KEY'])}",
    ])


def _phase_code(env: str, body: str) -> str:
    """
    Wrap phase body in a try/except that always prints a sentinel JSON line.
    This guarantees _parse_result() always finds the output even on error.
    """
    return f"""{env}

try:
{_indent(body)}
    print({repr(_SENTINEL)} + _result, flush=True)
except Exception as _e:
    import traceback as _tb
    print({repr(_SENTINEL)} + json.dumps({{"error": str(_e), "traceback": _tb.format_exc()}}), flush=True)
"""


def _indent(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _parse_result(raw: str) -> dict:
    """Extract and parse the sentinel-wrapped JSON from code_run stdout."""
    for line in raw.splitlines():
        if line.startswith(_SENTINEL):
            payload = json.loads(line[len(_SENTINEL):])
            if "error" in payload:
                raise RuntimeError(
                    f"Sandbox phase error: {payload['error']}\n"
                    f"{payload.get('traceback', '')}"
                )
            return payload
    raise ValueError(
        f"Sentinel '{_SENTINEL}' not found in sandbox output.\n"
        f"Raw (first 600 chars):\n{raw[:600]}"
    )


# ─── Phase code blocks ────────────────────────────────────────────────────────

def _recon_code(req: RedTeamRequest, env: str) -> str:
    body = f"""
from app.agents.recon_agent import run_recon

async def _run():
    recon = await run_recon({repr(req.target)})
    return recon.model_dump_json()

_result = asyncio.run(_run())
"""
    return _phase_code(env, body.strip())


def _plan_code(req: RedTeamRequest, recon_data: dict, env: str) -> str:
    import json as _json
    audit_cfg_json = _json.dumps(req.audit_config.model_dump(mode="json") if req.audit_config else None)
    recon_json = _json.dumps(recon_data)
    body = f"""
import json as _json
from app.agents.planner_agent import run_planning
from app.models.schemas import ReconReport, Depth, AuditConfig

async def _run():
    recon = ReconReport.model_validate(_json.loads({repr(recon_json)}))
    _cfg_raw = _json.loads({repr(audit_cfg_json)})
    audit_config = AuditConfig.model_validate(_cfg_raw) if _cfg_raw is not None else None
    plan = await run_planning(recon, Depth({repr(req.depth.value)}), audit_config)
    return plan.model_dump_json()

_result = asyncio.run(_run())
"""
    return _phase_code(env, body.strip())


def _suite_code(
    req: RedTeamRequest,
    suite_data: dict,
    recon_data: dict,
    env: str,
) -> str:
    import json as _json
    audit_cfg_json = _json.dumps(req.audit_config.model_dump(mode="json") if req.audit_config else None)
    recon_json = _json.dumps(recon_data)
    suite_json = _json.dumps(suite_data)
    body = f"""
import json as _json
from app.agents.attacker_agent import generate_probes
from app.agents.executor_agent import execute_probes_batch
from app.agents.judge_agent import judge_suite
from app.models.schemas import TestSuite, ReconReport, AuditConfig

_SUITE_TIMEOUT = 180  # 3 minutes — leaves 300s cleanup headroom within _acode_run's 480s max_wait

async def _run():
    suite = TestSuite.model_validate(_json.loads({repr(suite_json)}))
    recon = ReconReport.model_validate(_json.loads({repr(recon_json)}))
    _cfg_raw = _json.loads({repr(audit_cfg_json)})
    audit_config = AuditConfig.model_validate(_cfg_raw) if _cfg_raw is not None else None

    probes = await generate_probes(suite, recon, audit_config)
    pairs = await execute_probes_batch(
        probes,
        target={repr(req.target)},
        model_id={repr(req.resolved_model_id)},
        api_key={repr(req.target_api_key)},
        endpoint={repr(req.target_endpoint)},
    )
    result = await judge_suite(suite, pairs)
    return result.model_dump_json()

_result = asyncio.run(asyncio.wait_for(_run(), timeout=_SUITE_TIMEOUT))
"""
    return _phase_code(env, body.strip())


def _report_code(
    req: RedTeamRequest,
    recon_data: dict,
    plan_data: dict,
    suite_results: list[dict],
    env: str,
) -> str:
    body = f"""
from app.utils.report_builder import build_report
from app.models.schemas import ReconReport, RedTeamPlan, SuiteResult, PhasesCompleted

async def _run():
    recon        = ReconReport.model_validate({repr(recon_data)})
    plan         = RedTeamPlan.model_validate({repr(plan_data)})
    suite_res    = [SuiteResult.model_validate(s) for s in {repr(suite_results)}]
    phases       = PhasesCompleted(recon=True, planning=True, execution=bool(suite_res))

    report = await build_report(
        target={repr(req.target)},
        recon=recon,
        plan=plan,
        suite_results=suite_res,
        phases=phases,
    )
    return report.model_dump_json()

_result = asyncio.run(_run())
"""
    return _phase_code(env, body.strip())


# ─── SSE helper ───────────────────────────────────────────────────────────────

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ─── Worker sandbox helpers ───────────────────────────────────────────────────

async def _provision_worker(req: RedTeamRequest) -> tuple:
    """
    Create and fully bootstrap a worker sandbox.
    Returns (daytona_client, sandbox) — caller must tear both down.
    Bootstrap runs silently; logs show progress.
    """
    daytona = _make_client()
    sandbox = await daytona.create(CreateSandboxFromSnapshotParams(language="python"))
    logger.info("Worker sandbox created: id=%s", getattr(sandbox, "id", "?"))
    async for _ in _bootstrap(sandbox, install_deps=False):
        pass  # bootstrap progress consumed silently for workers (no pip install)
    logger.info("Worker sandbox ready: id=%s", getattr(sandbox, "id", "?"))
    return daytona, sandbox


async def _teardown_worker(daytona, sandbox) -> None:
    try:
        await daytona.delete(sandbox)
        logger.info("Worker sandbox removed: id=%s", getattr(sandbox, "id", "?"))
    except Exception as exc:
        logger.warning("Could not remove worker sandbox: %s", exc)
    try:
        await daytona.close()
    except Exception:
        pass


async def _run_suite_on_worker(
    req: RedTeamRequest,
    suite_data: dict,
    recon_data: dict,
    daytona,
    sandbox,
) -> dict:
    """Run one suite on a pre-bootstrapped worker sandbox and return result dict."""
    env = _env_setup(req)
    # max_wait=480: suite inner timeout=180s + up to 300s for httpx cleanup on cancellation
    raw = await _acode_run(sandbox, _suite_code(req, suite_data, recon_data, env), max_wait=480)
    return _parse_result(raw)


# ─── Main pipeline runner ─────────────────────────────────────────────────────

async def run_pipeline_in_sandbox(req: RedTeamRequest) -> AsyncIterator[str]:
    """
    Orchestrator: coordinates the 4-phase pipeline, streams SSE events.

    Orchestrator sandbox  → recon + planning + report
    Worker sandboxes      → one per suite (pre-spun during recon so bootstrap
                            cost is hidden inside the recon wait time)
    """
    daytona = _make_client()
    orch_sandbox = None
    worker_pool: list[tuple] = []       # list of (daytona_client, sandbox)
    worker_tasks: list[asyncio.Task] = []  # tracked so finally can cancel + teardown

    try:
        # ── Provision orchestrator sandbox ────────────────────────────────────
        yield _sse({"phase": "sandbox", "status": "provisioning"})
        orch_sandbox = await daytona.create(CreateSandboxFromSnapshotParams(language="python"))
        logger.info("Orchestrator sandbox created: id=%s", getattr(orch_sandbox, "id", "?"))

        # Bootstrap orchestrator + pre-spin worker sandboxes IN PARALLEL.
        # Worker bootstrap (~30-60s) overlaps with recon so users don't wait twice.
        _depth_counts = {"quick": 3, "standard": 6, "deep": 10}
        n_workers = _depth_counts.get(req.depth.value, 3)
        yield _sse({"phase": "sandbox", "status": "bootstrapping",
                    "message": f"Bootstrapping orchestrator + {n_workers} worker sandboxes…"})

        async def _bootstrap_orch():
            results = []
            async for p in _bootstrap(orch_sandbox):
                results.append(p)
            return results

        orch_bootstrap_task = asyncio.ensure_future(_bootstrap_orch())
        worker_tasks[:] = [asyncio.ensure_future(_provision_worker(req)) for _ in range(n_workers)]

        # Stream orchestrator bootstrap progress while workers spin up silently
        bootstrap_results = await orch_bootstrap_task
        for p in bootstrap_results:
            yield _sse(p)

        # Collect worker sandboxes (they may already be done)
        worker_pool = list(await asyncio.gather(*worker_tasks))
        yield _sse({"phase": "sandbox", "status": "ready",
                    "message": f"Orchestrator + {n_workers} workers ready"})

        env = _env_setup(req)

        # ── Phase 1: Recon ────────────────────────────────────────────────────
        yield _sse({"phase": "recon", "status": "started"})
        async for tick in _with_heartbeat(_acode_run(orch_sandbox, _recon_code(req, env)), "recon"):
            if isinstance(tick, str):
                raw = tick
            else:
                yield _sse(tick)
        recon_data = _parse_result(raw)
        yield _sse({"phase": "recon", "status": "complete", "data": recon_data})

        # ── Phase 2: Planning ─────────────────────────────────────────────────
        yield _sse({"phase": "planning", "status": "started"})
        async for tick in _with_heartbeat(_acode_run(orch_sandbox, _plan_code(req, recon_data, env)), "planning"):
            if isinstance(tick, str):
                raw = tick
            else:
                yield _sse(tick)
        plan_data = _parse_result(raw)
        yield _sse({"phase": "planning", "status": "complete", "data": plan_data})

        # ── Phase 3: Execution — dispatch suites to pre-warmed workers ────────
        suites = plan_data.get("test_suites", [])
        yield _sse({"phase": "execution", "status": "started", "total_suites": len(suites)})

        suite_results: list[dict] = []

        # Track which worker (daytona, sandbox) belongs to each future so we
        # can tear it down immediately when its suite completes — freeing disk ASAP.
        # overflow suites run on the orchestrator sandbox (no dedicated worker to tear down).
        suite_worker_pairs = list(zip(suites, worker_pool))
        overflow_suites = suites[len(worker_pool):]
        # worker_pool entries that didn't get a suite (depth > actual suites generated)
        extra_workers = worker_pool[len(suites):]

        # Tear down extra workers immediately — they won't be used
        for w_day, w_sb in extra_workers:
            asyncio.ensure_future(_teardown_worker(w_day, w_sb))
        worker_pool = worker_pool[:len(suites)]  # keep only what we're using

        suite_futures: dict[asyncio.Future, dict] = {}
        fut_to_worker: dict[asyncio.Future, tuple | None] = {}  # None = orch sandbox (don't teardown)

        for suite, (w_daytona, w_sandbox) in suite_worker_pairs:
            fut = asyncio.ensure_future(
                _run_suite_on_worker(req, suite, recon_data, w_daytona, w_sandbox)
            )
            suite_futures[fut] = suite
            fut_to_worker[fut] = (w_daytona, w_sandbox)

        for suite in overflow_suites:
            fut = asyncio.ensure_future(
                _run_suite_on_worker(req, suite, recon_data, daytona, orch_sandbox)
            )
            suite_futures[fut] = suite
            fut_to_worker[fut] = None  # orchestrator sandbox — don't tear down mid-pipeline

        pending = set(suite_futures.keys())
        exec_elapsed = 0
        rate_limit_detected = False

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED, timeout=10.0
            )
            if not done:
                exec_elapsed += 10
                if exec_elapsed >= 60 and not rate_limit_detected:
                    rate_limit_detected = True
                    yield _sse({
                        "phase": "execution", "status": "rate_limited",
                        "message": (
                            f"Rate limiting detected on target model — retrying automatically… "
                            f"({exec_elapsed}s elapsed, {len(pending)} suites remaining)"
                        ),
                    })
                elif rate_limit_detected:
                    yield _sse({
                        "phase": "execution", "status": "rate_limited",
                        "message": f"Still retrying after rate limit… ({exec_elapsed}s elapsed, {len(pending)} suites remaining)",
                    })
                else:
                    yield _sse({"phase": "execution", "status": "running",
                                 "message": f"Suites running… ({exec_elapsed}s elapsed, {len(pending)} remaining)"})

            for fut in done:
                suite = suite_futures[fut]
                # Tear down this suite's worker immediately to free disk space
                worker_pair = fut_to_worker.get(fut)
                if worker_pair is not None:
                    asyncio.ensure_future(_teardown_worker(*worker_pair))
                    # Remove from worker_pool so finally block doesn't double-teardown
                    if worker_pair in worker_pool:
                        worker_pool.remove(worker_pair)
                try:
                    result = fut.result()
                    suite_results.append(result)
                    yield _sse({
                        "phase": "execution",
                        "suite": result.get("suite_name"),
                        "status": "complete",
                        "data": result,
                    })
                except Exception as exc:
                    err_str = str(exc)
                    is_timeout = "TimeoutError" in err_str or "timed out" in err_str.lower()
                    logger.error("Suite '%s' failed: %s", suite.get("suite_name"), exc)
                    yield _sse({
                        "phase": "execution",
                        "suite": suite.get("suite_name"),
                        "status": "error",
                        "error": (
                            "Suite timed out — likely caused by API rate limiting. "
                            "Consider using a higher-quota API key."
                        ) if is_timeout else err_str,
                        "rate_limited": is_timeout,
                    })

        # ── Phase 4: Report (on orchestrator sandbox) ─────────────────────────
        yield _sse({"phase": "report", "status": "started"})
        raw = await _acode_run(
            orch_sandbox,
            _report_code(req, recon_data, plan_data, suite_results, env),
        )
        report_data = _parse_result(raw)
        yield _sse({"phase": "report", "status": "complete", "data": report_data})

    except Exception as exc:
        logger.error("Pipeline fatal error: %s", exc, exc_info=True)
        yield _sse({"phase": "error", "status": "error", "error": str(exc)})

    finally:
        # ── Cancel any in-flight worker provisioning tasks ─────────────────────
        # If an exception fires before asyncio.gather() resolves, worker sandboxes
        # may have been created but not yet returned into worker_pool — cancel the
        # tasks and collect whatever completed so we can delete those sandboxes too.
        for t in worker_tasks:
            if not t.done():
                t.cancel()
        for t in worker_tasks:
            try:
                result = await t  # may raise CancelledError or provision error
                if result not in worker_pool:
                    worker_pool.append(result)
            except Exception:
                pass  # task cancelled or provisioning failed — nothing to teardown

        # ── Tear down orchestrator sandbox ─────────────────────────────────────
        if orch_sandbox is not None:
            try:
                await daytona.delete(orch_sandbox)
                logger.info("Orchestrator sandbox removed: id=%s", getattr(orch_sandbox, "id", "?"))
            except Exception as exc:
                logger.warning("Could not remove orchestrator sandbox: %s", exc)
        try:
            await daytona.close()
        except Exception:
            pass
        # ── Tear down all remaining worker sandboxes ───────────────────────────
        for w_daytona, w_sandbox in worker_pool:
            await _teardown_worker(w_daytona, w_sandbox)
        logger.info("Cleanup complete — all sandboxes removed.")

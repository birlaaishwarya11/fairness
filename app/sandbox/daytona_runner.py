"""
daytona_runner.py

Executes the full FairSight pipeline inside an ephemeral Daytona sandbox.

Each POST /red-team request:
  1. Creates a fresh Python sandbox via the Daytona API
  2. Uploads all project .py source files to /workspace/
  3. Installs Python dependencies
  4. Runs each pipeline phase as an isolated code_run() call
  5. Yields phase events as SSE-formatted strings (real-time streaming)
  6. Destroys the sandbox on completion or error

Phase streaming works by executing one code_run per phase/suite — each call
blocks until that unit of work completes, then we yield the SSE event and
move to the next. Suite execution is parallelised across threads using
asyncio.get_event_loop().run_in_executor so multiple suites run concurrently
while still yielding individual completion events as they arrive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxParams

from app.models.schemas import FinalReport, RedTeamRequest

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # .../fairsight-backend/
_WORKSPACE = "/workspace"
_SENTINEL = "FAIRSIGHT_RESULT:"

# Pinned versions mirror requirements.txt (minus the server-side packages)
_SANDBOX_DEPS = [
    "anthropic==0.28.0",
    "tavily-python==0.3.3",
    "httpx==0.27.0",
    "pydantic>=2.9.0",
    "python-dotenv==1.0.1",
]


# ─── Daytona client ───────────────────────────────────────────────────────────

def _make_client() -> Daytona:
    return Daytona(
        DaytonaConfig(
            api_key=os.environ["DAYTONA_API_KEY"],
            server_url=os.environ.get(
                "DAYTONA_SERVER_URL", "https://app.daytona.io/api"
            ),
        )
    )


# ─── Async wrappers for the sync Daytona SDK ─────────────────────────────────

async def _aexec(sandbox, cmd: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: sandbox.process.exec(cmd)
    )
    return getattr(result, "output", "") or ""


async def _acode_run(sandbox, code: str) -> str:
    """Run Python code in the sandbox; return raw stdout."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: sandbox.process.code_run(code)
    )
    return getattr(result, "result", "") or ""


async def _aupload(sandbox, dest_path: str, content: bytes) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: sandbox.fs.upload_file(dest_path, content)
    )


# ─── Sandbox bootstrap ────────────────────────────────────────────────────────

async def _bootstrap(sandbox) -> None:
    """
    Install deps + upload all project Python source files in parallel.
    This is done once per sandbox, before any phase runs.
    """
    # Collect all .py files (skip __pycache__)
    py_files = [
        p for p in _PROJECT_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    ]

    # Create directory tree first
    dirs = {
        str(Path(p.relative_to(_PROJECT_ROOT)).parent)
        for p in py_files
    }
    mkdir_tasks = [
        _aexec(sandbox, f"mkdir -p {_WORKSPACE}/{d}")
        for d in sorted(dirs)
        if d != "."
    ]
    await asyncio.gather(*mkdir_tasks)

    # Upload files + install deps in parallel
    upload_tasks = [
        _aupload(sandbox, f"{_WORKSPACE}/{p.relative_to(_PROJECT_ROOT)}", p.read_bytes())
        for p in py_files
    ]
    install_task = _aexec(
        sandbox,
        f"pip install -q {' '.join(_SANDBOX_DEPS)}"
    )
    await asyncio.gather(*upload_tasks, install_task)
    logger.info("Sandbox bootstrapped: %d files, deps installed", len(py_files))


# ─── Code template helpers ────────────────────────────────────────────────────

def _env_setup(req: RedTeamRequest) -> str:
    """
    Python snippet injected at the top of every phase code block.
    Sets env vars and sys.path so imports work inside the sandbox.

    FairSight's internal agents read FAIRSIGHT_API_KEY / FAIRSIGHT_MODEL /
    FAIRSIGHT_ENDPOINT — sourced entirely from the request, not server env.
    TAVILY_API_KEY is the only server-side key (recon web search).
    """
    return "\n".join([
        "import os, sys, json, asyncio",
        f"sys.path.insert(0, {repr(_WORKSPACE)})",
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
    body = f"""
from app.agents.planner_agent import run_planning
from app.models.schemas import ReconReport, Depth

async def _run():
    recon = ReconReport.model_validate({repr(recon_data)})
    plan = await run_planning(recon, Depth({repr(req.depth.value)}))
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
    body = f"""
from app.agents.attacker_agent import generate_probes
from app.agents.executor_agent import execute_probes_batch
from app.agents.judge_agent import judge_suite
from app.models.schemas import TestSuite, ReconReport

async def _run():
    suite = TestSuite.model_validate({repr(suite_data)})
    recon = ReconReport.model_validate({repr(recon_data)})

    probes = await generate_probes(suite, recon)
    pairs = await execute_probes_batch(
        probes,
        target={repr(req.target)},
        model_id={repr(req.resolved_model_id)},
        api_key={repr(req.target_api_key)},
        endpoint={repr(req.target_endpoint)},
    )
    result = await judge_suite(suite, pairs)
    return result.model_dump_json()

_result = asyncio.run(_run())
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


# ─── Main pipeline runner ─────────────────────────────────────────────────────

async def run_pipeline_in_sandbox(req: RedTeamRequest) -> AsyncIterator[str]:
    """
    Run the full red-team pipeline inside a Daytona sandbox.
    Yields SSE-formatted strings as each phase completes.
    The sandbox is always destroyed on exit (success or error).
    """
    daytona = _make_client()
    sandbox = None

    try:
        # ── Provision sandbox ─────────────────────────────────────────────────
        yield _sse({"phase": "sandbox", "status": "provisioning"})

        loop = asyncio.get_event_loop()
        sandbox = await loop.run_in_executor(
            None,
            lambda: daytona.create(CreateSandboxParams(language="python")),
        )
        logger.info("Sandbox created: id=%s", getattr(sandbox, "id", "?"))

        yield _sse({"phase": "sandbox", "status": "bootstrapping"})
        await _bootstrap(sandbox)
        yield _sse({"phase": "sandbox", "status": "ready"})

        env = _env_setup(req)

        # ── Phase 1: Recon ────────────────────────────────────────────────────
        yield _sse({"phase": "recon", "status": "started"})
        raw = await _acode_run(sandbox, _recon_code(req, env))
        recon_data = _parse_result(raw)
        yield _sse({"phase": "recon", "status": "complete", "data": recon_data})

        # ── Phase 2: Planning ─────────────────────────────────────────────────
        yield _sse({"phase": "planning", "status": "started"})
        raw = await _acode_run(sandbox, _plan_code(req, recon_data, env))
        plan_data = _parse_result(raw)
        yield _sse({"phase": "planning", "status": "complete", "data": plan_data})

        # ── Phase 3: Execution (suites run in parallel threads) ───────────────
        suites = plan_data.get("test_suites", [])
        yield _sse({"phase": "execution", "status": "started", "total_suites": len(suites)})

        suite_results: list[dict] = []

        # Build one Future per suite so we can stream completions as they arrive
        suite_futures: dict[asyncio.Future, dict] = {
            asyncio.ensure_future(
                _acode_run(sandbox, _suite_code(req, suite, recon_data, env))
            ): suite
            for suite in suites
        }

        pending = set(suite_futures.keys())
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                suite = suite_futures[fut]
                try:
                    raw = fut.result()
                    result = _parse_result(raw)
                    suite_results.append(result)
                    yield _sse({
                        "phase": "execution",
                        "suite": result.get("suite_name"),
                        "status": "complete",
                        "data": result,
                    })
                except Exception as exc:
                    logger.error(
                        "Suite '%s' failed in sandbox: %s",
                        suite.get("suite_name"), exc,
                    )
                    yield _sse({
                        "phase": "execution",
                        "suite": suite.get("suite_name"),
                        "status": "error",
                        "error": str(exc),
                    })

        # ── Phase 4: Report ───────────────────────────────────────────────────
        yield _sse({"phase": "report", "status": "started"})
        raw = await _acode_run(
            sandbox,
            _report_code(req, recon_data, plan_data, suite_results, env),
        )
        report_data = _parse_result(raw)
        yield _sse({"phase": "report", "status": "complete", "data": report_data})

    except Exception as exc:
        logger.error("Sandbox pipeline fatal error: %s", exc, exc_info=True)
        yield _sse({"phase": "error", "status": "error", "error": str(exc)})

    finally:
        if sandbox is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: daytona.remove(sandbox))
                logger.info("Sandbox removed: id=%s", getattr(sandbox, "id", "?"))
            except Exception as exc:
                logger.warning("Could not remove sandbox: %s", exc)

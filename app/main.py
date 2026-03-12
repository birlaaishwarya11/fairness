"""
main.py
FairSight FastAPI application.

Every POST /red-team request spins up a fresh Daytona sandbox, runs the
full agentic pipeline inside it, streams SSE events back to the client,
then destroys the sandbox.  The FastAPI process itself never imports the
agent modules — all AI work happens in the isolated sandbox.

Routes:
  POST /red-team    → SSE stream (sandbox lifecycle + pipeline phases)
  GET  /report/{id} → stored FinalReport JSON
  GET  /health      → health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models.schemas import FinalReport, RedTeamRequest
from app.sandbox.daytona_runner import run_pipeline_in_sandbox, _make_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── In-memory report store ────────────────────────────────────────────────────
_reports: dict[str, FinalReport] = {}

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FairSight — AI Red Teaming & Fairness Auditing API",
    version="1.0.0",
    description=(
        "Agentic red-team pipeline executed inside isolated Daytona sandboxes. "
        "Audits AI systems for bias, jailbreaks, hallucinations, PII leakage, "
        "toxicity, and demographic gaps."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pipeline stream ──────────────────────────────────────────────────────────

_SSE_KEEPALIVE = ": ping\n\n"  # SSE comment — invisible to app, resets all timeouts
_KEEPALIVE_INTERVAL = 5        # seconds between pings


async def _stream_pipeline(req: RedTeamRequest):
    """
    Wraps the sandbox pipeline with a background keepalive task.

    A separate coroutine sends `: ping` SSE comment lines every 5 seconds
    so the connection never goes silent long enough for Lovable (or any
    intermediate proxy) to declare it dead.  Pipeline events are forwarded
    via an asyncio.Queue so both producers share one consumer loop.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _pipeline_producer() -> None:
        try:
            async for sse_str in run_pipeline_in_sandbox(req):
                # Persist the final report server-side
                if sse_str.startswith("data: "):
                    try:
                        event = json.loads(sse_str[6:])
                        if event.get("phase") == "report" and event.get("status") == "complete":
                            report = FinalReport.model_validate(event["data"])
                            _reports[report.report_id] = report
                            logger.info("Stored report %s (target=%s)", report.report_id, req.target)
                    except Exception:
                        pass
                await queue.put(sse_str)
        except Exception as exc:
            logger.error("Pipeline producer error: %s", exc)
        finally:
            await queue.put(None)  # sentinel — tells consumer to stop

    async def _keepalive_producer() -> None:
        while True:
            await asyncio.sleep(_KEEPALIVE_INTERVAL)
            await queue.put(_SSE_KEEPALIVE)

    pipeline_task = asyncio.create_task(_pipeline_producer())
    keepalive_task = asyncio.create_task(_keepalive_producer())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        pipeline_task.cancel()
        keepalive_task.cancel()
        # Drain exceptions from cancelled tasks silently
        for t in (pipeline_task, keepalive_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/red-team", summary="Start a red-team audit (SSE stream)")
async def red_team(req: RedTeamRequest):
    """
    Provisions a Daytona sandbox, runs the full pipeline inside it,
    and streams phase-by-phase Server-Sent Events back to the caller.

    SSE event sequence:
      sandbox/provisioning → sandbox/bootstrapping → sandbox/ready
      recon/started        → recon/complete
      planning/started     → planning/complete
      execution/started    → execution/complete (×N suites)
      report/started       → report/complete
    """
    return StreamingResponse(
        _stream_pipeline(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/report/{report_id}", response_model=FinalReport, summary="Retrieve a stored report")
async def get_report(report_id: str):
    """Return a previously generated FinalReport by its UUID."""
    report = _reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return report


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "reports_in_memory": len(_reports),
    }


# ─── Admin: sandbox cleanup ────────────────────────────────────────────────────

@app.get("/admin/sandboxes", summary="List all active Daytona sandboxes")
async def list_sandboxes():
    """List all sandboxes in your Daytona account with their IDs and state."""
    daytona = _make_client()
    try:
        sandboxes = await daytona.list()
        return {
            "count": len(sandboxes),
            "sandboxes": [
                {
                    "id": getattr(s, "id", "?"),
                    "state": getattr(s, "state", "?"),
                    "created_at": str(getattr(s, "created_at", "?")),
                }
                for s in sandboxes
            ],
        }
    finally:
        try:
            await daytona.close()
        except Exception:
            pass


@app.delete("/admin/sandboxes", summary="Delete ALL active Daytona sandboxes")
async def delete_all_sandboxes():
    """
    Bulk-delete every sandbox in your Daytona account.
    Use this to recover disk space after failed or stuck runs.
    """
    daytona = _make_client()
    deleted, failed = 0, 0
    try:
        sandboxes = await daytona.list()
        for s in sandboxes:
            sid = getattr(s, "id", None)
            try:
                await daytona.delete(s)
                logger.info("Admin: deleted sandbox %s", sid)
                deleted += 1
            except Exception as exc:
                logger.warning("Admin: could not delete sandbox %s: %s", sid, exc)
                failed += 1
        return {"deleted": deleted, "failed": failed, "total": len(sandboxes)}
    finally:
        try:
            await daytona.close()
        except Exception:
            pass


@app.delete("/admin/sandboxes/{sandbox_id}", summary="Delete a specific Daytona sandbox")
async def delete_sandbox(sandbox_id: str):
    """Delete a single sandbox by ID."""
    daytona = _make_client()
    try:
        sandboxes = await daytona.list()
        target = next((s for s in sandboxes if getattr(s, "id", None) == sandbox_id), None)
        if not target:
            raise HTTPException(status_code=404, detail=f"Sandbox '{sandbox_id}' not found.")
        await daytona.delete(target)
        logger.info("Admin: deleted sandbox %s", sandbox_id)
        return {"deleted": sandbox_id}
    finally:
        try:
            await daytona.close()
        except Exception:
            pass

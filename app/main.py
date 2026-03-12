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

import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.models.schemas import FinalReport, RedTeamRequest
from app.sandbox.daytona_runner import run_pipeline_in_sandbox

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

async def _stream_pipeline(req: RedTeamRequest):
    """
    Async generator that delegates to the Daytona sandbox runner.
    Intercepts the final 'report/complete' event to persist the report,
    then re-yields it so the client receives it unchanged.
    """
    async for sse_str in run_pipeline_in_sandbox(req):
        # Check for the final report so we can store it server-side
        if sse_str.startswith("data: "):
            try:
                event = json.loads(sse_str[6:])
                if event.get("phase") == "report" and event.get("status") == "complete":
                    report = FinalReport.model_validate(event["data"])
                    _reports[report.report_id] = report
                    logger.info(
                        "Stored report %s (target=%s)", report.report_id, req.target
                    )
            except Exception:
                pass  # never block the stream on a storage error

        yield sse_str


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

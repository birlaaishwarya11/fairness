"""
test_pipeline.py
Validates the FairSight SSE streaming pipeline end-to-end.

Usage:
    python test_pipeline.py [--target "GPT-4o"] [--depth quick]

Requires the server to be running:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional

import httpx

BASE_URL = "http://localhost:8000"

REQUIRED_REPORT_FIELDS = {
    "report_id",
    "target",
    "generated_at",
    "recon_summary",
    "plan_summary",
    "overall_risk_score",
    "overall_risk_level",
    "executive_summary",
    "suite_results",
    "top_findings",
    "recommendations",
    "phases_completed",
}

REQUIRED_SUITE_FIELDS = {
    "suite_name",
    "risk_level",
    "average_score",
    "probes_run",
    "probes_failed",
    "findings",
    "suite_summary",
}


# ─── Colours ──────────────────────────────────────────────────────────────────

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str: return _c("31", t)
def bold(t: str) -> str: return _c("1", t)
def cyan(t: str) -> str: return _c("36", t)
def dim(t: str) -> str: return _c("2", t)


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_report(report: dict) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_REPORT_FIELDS - set(report.keys())
    if missing:
        errors.append(f"Missing top-level fields: {', '.join(sorted(missing))}")

    score = report.get("overall_risk_score")
    if score is not None and not (0 <= float(score) <= 100):
        errors.append(f"overall_risk_score out of range: {score}")

    suites = report.get("suite_results", [])
    if not isinstance(suites, list):
        errors.append("suite_results is not a list")
    else:
        for i, suite in enumerate(suites):
            suite_missing = REQUIRED_SUITE_FIELDS - set(suite.keys())
            if suite_missing:
                errors.append(
                    f"suite_results[{i}] missing fields: {', '.join(sorted(suite_missing))}"
                )

    phases = report.get("phases_completed", {})
    if not isinstance(phases, dict):
        errors.append("phases_completed is not a dict")

    return errors


# ─── Health Check ─────────────────────────────────────────────────────────────

def check_health(client: httpx.Client) -> bool:
    print(f"\n{bold('=== Health Check ===')}")
    try:
        resp = client.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        print(f"  Status:  {green(data.get('status', '?'))}")
        print(f"  Version: {data.get('version', '?')}")
        return True
    except Exception as exc:
        print(f"  {red(f'Health check FAILED: {exc}')}")
        print(f"  {yellow('Is the server running? uvicorn app.main:app --reload')}")
        return False


# ─── SSE Stream Consumer ──────────────────────────────────────────────────────

def consume_stream(
    client: httpx.Client,
    model_name: str,
    model_version: Optional[str],
    api_key: str,
    depth: str,
    target_endpoint: Optional[str] = None,
    timeout: int = 600,
) -> Optional[dict]:
    """
    POST to /red-team and consume the SSE stream.
    Prints progress as each event arrives.
    Returns the final report dict, or None if something went wrong.
    """
    payload: dict = {
        "model_name": model_name,
        "target_api_key": api_key,
        "depth": depth,
    }
    if model_version:
        payload["model_version"] = model_version
    if target_endpoint:
        payload["target_endpoint"] = target_endpoint

    display_target = f"{model_name}" + (f" ({model_version})" if model_version else "")
    phases_seen: list[str] = []
    suites_complete: list[str] = []
    final_report: Optional[dict] = None
    buffer = ""

    print(f"\n{bold('=== Starting Red-Team Pipeline ===')}")
    print(f"  Model:    {cyan(display_target)}")
    print(f"  Model ID: {cyan(model_name + ('-' + model_version if model_version else ''))}")
    print(f"  Depth:    {cyan(depth)}")
    print()

    t0 = time.time()

    try:
        with client.stream(
            "POST",
            f"{BASE_URL}/red-team",
            json=payload,
            timeout=timeout,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue

                raw = line[len("data:"):].strip()
                try:
                    event: dict = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"  {yellow('Non-JSON SSE line:')} {raw[:120]}")
                    continue

                phase = event.get("phase", "?")
                status = event.get("status", "?")
                elapsed = time.time() - t0

                _handle_event(event, phase, status, elapsed, phases_seen, suites_complete)

                if phase == "report" and status == "complete":
                    final_report = event.get("data")
                    break

                if status == "error":
                    print(f"\n  {red('Error in phase ' + phase + ':')} {event.get('error')}")
                    break

    except httpx.ConnectError:
        print(f"\n{red('Connection refused.')} Is the server running at {BASE_URL}?")
    except httpx.ReadTimeout:
        print(f"\n{red('Stream timed out after')} {timeout}s")
    except Exception as exc:
        print(f"\n{red('Unexpected error:')} {exc}")

    return final_report


def _handle_event(
    event: dict,
    phase: str,
    status: str,
    elapsed: float,
    phases_seen: list[str],
    suites_complete: list[str],
) -> None:
    ts = f"[{elapsed:6.1f}s]"

    if phase == "recon":
        if status == "started":
            phases_seen.append("recon")
            print(f"  {dim(ts)} {bold('RECON')}      → {yellow('started')}")
        elif status == "complete":
            data = event.get("data", {})
            findings_count = (
                len(data.get("known_vulnerabilities", []))
                + len(data.get("academic_critiques", []))
                + len(data.get("regulatory_exposure", []))
                + len(data.get("demographic_gaps", []))
            )
            models = data.get("detected_models", [])
            print(f"  {dim(ts)} {bold('RECON')}      → {green('complete')}  "
                  f"({findings_count} findings, models: {models or 'unknown'})")
            if data.get("most_concerning"):
                print(f"             {dim('Most concerning:')} {data['most_concerning'][:100]}")

    elif phase == "planning":
        if status == "started":
            phases_seen.append("planning")
            print(f"  {dim(ts)} {bold('PLANNING')}   → {yellow('started')}")
        elif status == "complete":
            data = event.get("data", {})
            n = len(data.get("test_suites", []))
            print(f"  {dim(ts)} {bold('PLANNING')}   → {green('complete')}  ({n} suites planned)")
            for s in data.get("test_suites", []):
                print(f"             {dim('•')} {s.get('suite_name')} "
                      f"[{s.get('probe_category')}] "
                      f"× {s.get('num_probes')} probes")

    elif phase == "execution":
        if status == "started":
            phases_seen.append("execution")
            total = event.get("total_suites", "?")
            print(f"\n  {dim(ts)} {bold('EXECUTION')}  → {yellow('started')}  ({total} suites)")
        elif status == "complete":
            suite_name = event.get("suite", "?")
            data = event.get("data", {})
            risk = data.get("risk_level", "?")
            avg = data.get("average_score", "?")
            failed = data.get("probes_failed", "?")
            total_p = data.get("probes_run", "?")
            suites_complete.append(suite_name)
            risk_colour = red if risk in ("CRITICAL", "HIGH") else (yellow if risk == "MEDIUM" else green)
            print(f"  {dim(ts)}   {dim('suite')} {suite_name[:50]:<50} "
                  f"{risk_colour(risk):<8} score={avg} fail={failed}/{total_p}")

    elif phase == "report":
        if status == "started":
            print(f"\n  {dim(ts)} {bold('REPORT')}     → {yellow('building')}...")
        elif status == "complete":
            data = event.get("data", {})
            score = data.get("overall_risk_score", "?")
            level = data.get("overall_risk_level", "?")
            level_colour = red if level in ("CRITICAL", "HIGH") else (yellow if level == "MEDIUM" else green)
            print(f"  {dim(ts)} {bold('REPORT')}     → {green('complete')}  "
                  f"risk={level_colour(level)} score={score}/100")


# ─── Summary Printer ──────────────────────────────────────────────────────────

def print_summary(report: dict, errors: list[str]) -> None:
    print(f"\n{bold('=' * 60)}")
    print(bold("  FINAL REPORT SUMMARY"))
    print(bold("=" * 60))

    print(f"  Report ID:      {report.get('report_id', '?')}")
    print(f"  Target:         {report.get('target', '?')}")
    print(f"  Generated at:   {report.get('generated_at', '?')}")

    score = report.get("overall_risk_score", 0)
    level = report.get("overall_risk_level", "?")
    level_colour = red if level in ("CRITICAL", "HIGH") else (yellow if level == "MEDIUM" else green)
    print(f"  Risk score:     {level_colour(f'{score}/100  ({level})')}")

    phases = report.get("phases_completed", {})
    def tick(v: bool) -> str: return green("✓") if v else red("✗")
    print(f"\n  Phases completed:")
    print(f"    {tick(phases.get('recon', False))}  Recon")
    print(f"    {tick(phases.get('planning', False))}  Planning")
    print(f"    {tick(phases.get('execution', False))}  Execution")

    suites = report.get("suite_results", [])
    total_probes = sum(s.get("probes_run", 0) for s in suites)
    total_failed = sum(s.get("probes_failed", 0) for s in suites)
    print(f"\n  Suites run:     {len(suites)}")
    print(f"  Probes run:     {total_probes}")
    print(f"  Probes failed:  {total_failed}  ({total_failed/total_probes*100:.0f}% failure rate)" if total_probes else "")

    print(f"\n  {bold('Executive Summary:')}")
    summary = report.get("executive_summary", "N/A")
    for line in _wrap(summary, 56):
        print(f"    {line}")

    top = report.get("top_findings", [])
    if top:
        print(f"\n  {bold('Top Findings:')}")
        for i, finding in enumerate(top[:5], 1):
            for j, line in enumerate(_wrap(finding, 54)):
                prefix = f"  {i}." if j == 0 else "     "
                print(f"    {prefix} {line}")

    recs = report.get("recommendations", [])
    if recs:
        print(f"\n  {bold('Recommendations:')}")
        for rec in recs:
            for j, line in enumerate(_wrap(rec, 54)):
                prefix = "  •" if j == 0 else "   "
                print(f"    {prefix} {line}")

    print(f"\n{bold('=' * 60)}")
    if errors:
        print(f"  {red(bold('VALIDATION ERRORS:'))}")
        for err in errors:
            print(f"    {red('•')} {err}")
        print(f"{bold('=' * 60)}")
    else:
        print(f"  {green(bold('All required fields present — validation PASSED'))}")
        print(f"{bold('=' * 60)}")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            if current:
                lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines or [""]


# ─── Report Retrieval Test ────────────────────────────────────────────────────

def test_report_retrieval(client: httpx.Client, report_id: str) -> bool:
    print(f"\n{bold('=== Testing GET /report/{report_id} ===')}")
    try:
        resp = client.get(f"{BASE_URL}/report/{report_id}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("report_id") == report_id:
            print(f"  {green('✓')} Report retrieved successfully")
            return True
        else:
            print(f"  {red('✗')} report_id mismatch in response")
            return False
    except Exception as exc:
        print(f"  {red(f'✗ Failed: {exc}')}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FairSight pipeline integration test")
    parser.add_argument(
        "--model-name", default="gpt-4o",
        help="Target model name (e.g. gpt-4o, claude-opus-4-6, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--model-version", default=None,
        help="Optional model version/snapshot (e.g. 2024-11-20). Appended to model-name for the API call.",
    )
    parser.add_argument(
        "--api-key", required=True,
        help="API key for the target model",
    )
    parser.add_argument(
        "--target-endpoint", default=None,
        help="Custom API endpoint URL (optional, inferred from model-name if omitted)",
    )
    parser.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="quick",
        help="Audit depth (quick=3 suites, standard=6, deep=10+)",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="FairSight API base URL")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Stream timeout in seconds"
    )
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = args.base_url

    overall_pass = True

    with httpx.Client() as client:
        # 1. Health check
        if not check_health(client):
            sys.exit(1)

        # 2. Run full pipeline
        report = consume_stream(
            client,
            model_name=args.model_name,
            model_version=args.model_version,
            api_key=args.api_key,
            depth=args.depth,
            target_endpoint=args.target_endpoint,
            timeout=args.timeout,
        )

        if report is None:
            print(f"\n{red('Pipeline did not return a report — FAILED')}")
            sys.exit(1)

        # 3. Validate report structure
        errors = validate_report(report)

        # 4. Print summary
        print_summary(report, errors)

        # 5. Test report retrieval endpoint
        report_id = report.get("report_id")
        if report_id:
            retrieval_ok = test_report_retrieval(client, report_id)
            if not retrieval_ok:
                overall_pass = False
        else:
            print(f"\n{red('No report_id in response — skipping retrieval test')}")
            overall_pass = False

        if errors:
            overall_pass = False

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

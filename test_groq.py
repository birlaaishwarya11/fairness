#!/usr/bin/env python3
"""
test_groq.py
End-to-end test of the FairSight pipeline using Groq as both the
FairSight auditing engine and the target model under test.

Usage:
    python test_groq.py --groq-key <your-groq-key> [--url http://localhost:8000]
    python test_groq.py --groq-key <key> --save-md report.md

Best Groq models (as of early 2026):
  - llama-3.3-70b-versatile   (default — best quality, 128k context)
  - llama3-70b-8192           (fast, reliable)
  - deepseek-r1-distill-llama-70b  (strong reasoning)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_FAIRSIGHT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TARGET_MODEL = "llama-3.3-70b-versatile"

RISK_EMOJI = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FairSight Groq smoke test")
    p.add_argument("--groq-key", required=True, help="Your Groq API key")
    p.add_argument(
        "--url",
        default="http://localhost:8000",
        help="FairSight backend URL (default: http://localhost:8000)",
    )
    p.add_argument(
        "--fairsight-model",
        default=DEFAULT_FAIRSIGHT_MODEL,
        help=f"Groq model for FairSight agents (default: {DEFAULT_FAIRSIGHT_MODEL})",
    )
    p.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help=f"Groq model to audit (default: {DEFAULT_TARGET_MODEL})",
    )
    p.add_argument(
        "--depth",
        choices=["quick", "standard", "deep"],
        default="quick",
        help="Audit depth (default: quick)",
    )
    p.add_argument(
        "--save-md",
        metavar="FILE",
        help="Save the final report as a Markdown file (e.g. report.md)",
    )
    return p.parse_args()


# ── Markdown formatter ────────────────────────────────────────────────────────

def _md_report(report: dict, elapsed: float, target_model: str, depth: str) -> str:
    target = report.get("target", target_model)
    risk_level = report.get("overall_risk_level", "UNKNOWN")
    risk_score = report.get("overall_risk_score", 0)
    emoji = RISK_EMOJI.get(risk_level, "⚪")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# FairSight Red Team Report",
        f"",
        f"| Field | Value |",
        f"|---|---|",
        f"| **Target model** | `{target}` |",
        f"| **Audit depth** | {depth} |",
        f"| **Generated** | {generated_at} |",
        f"| **Duration** | {elapsed:.1f}s |",
        f"",
    ]

    # ── Overall risk ──────────────────────────────────────────────────────────
    lines += [
        f"## {emoji} Overall Risk: {risk_level} ({risk_score}/100)",
        f"",
    ]

    # ── Executive summary ─────────────────────────────────────────────────────
    summary = report.get("executive_summary", "").strip()
    if summary:
        lines += [
            f"## Executive Summary",
            f"",
            summary,
            f"",
        ]

    # ── Recon summary ─────────────────────────────────────────────────────────
    recon = report.get("recon_summary", "").strip()
    if recon:
        lines += [
            f"## Reconnaissance",
            f"",
            f"> {recon}",
            f"",
        ]

    # ── Suite results ─────────────────────────────────────────────────────────
    suite_results = report.get("suite_results", [])
    if suite_results:
        lines += [f"## Test Suite Results", f""]
        for sr in suite_results:
            sr_risk = sr.get("risk_level", "UNKNOWN")
            sr_emoji = RISK_EMOJI.get(sr_risk, "⚪")
            lines += [
                f"### {sr_emoji} {sr.get('suite_name')}",
                f"",
                f"| Metric | Value |",
                f"|---|---|",
                f"| Category | `{sr.get('probe_category')}` |",
                f"| Risk level | **{sr_risk}** |",
                f"| Avg safety score | {sr.get('average_score')}/10 |",
                f"| Probes failed | {sr.get('probes_failed')}/{sr.get('probes_run')} |",
                f"",
            ]
            summary_text = sr.get("suite_summary", "").strip()
            if summary_text:
                lines += [f"{summary_text}", f""]

            # Individual findings
            findings = sr.get("findings", [])
            failed = [f for f in findings if f.get("verdict") in ("FAIL", "PARTIAL")]
            if failed:
                lines += [f"**Key findings:**", f""]
                for f in failed[:5]:
                    verdict_icon = "❌" if f.get("verdict") == "FAIL" else "⚠️"
                    sev = f.get("severity", "")
                    reasoning = f.get("reasoning", "").strip()
                    evidence = f.get("evidence", "").strip()
                    lines += [
                        f"- {verdict_icon} **[{sev}]** {reasoning}",
                    ]
                    if evidence:
                        lines += [f"  > *Evidence:* `{evidence[:200]}`"]
                lines += [f""]

    # ── Top findings ──────────────────────────────────────────────────────────
    top = report.get("top_findings", [])
    if top:
        lines += [f"## Top Findings", f""]
        for i, finding in enumerate(top, 1):
            lines += [f"{i}. {finding}"]
        lines += [f""]

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = report.get("recommendations", [])
    if recs:
        lines += [f"## Recommendations", f""]
        for i, rec in enumerate(recs, 1):
            lines += [f"{i}. {rec}"]
        lines += [f""]

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        f"---",
        f"*Generated by [FairSight](https://github.com/birlaaishwarya11/fairness) — AI Red Teaming & Fairness Auditing*",
    ]

    return "\n".join(lines)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_test(args: argparse.Namespace) -> None:
    payload = {
        "model_name": args.target_model,
        "model_version": None,
        "target_api_key": args.groq_key,
        "target_endpoint": GROQ_ENDPOINT,
        "fairsight_api_key": args.groq_key,
        "fairsight_model": args.fairsight_model,
        "fairsight_endpoint": GROQ_ENDPOINT,
        "depth": args.depth,
    }

    print(f"\n{'='*60}")
    print(f"  FairSight Groq Test")
    print(f"  Target model : {args.target_model}")
    print(f"  Audit model  : {args.fairsight_model}")
    print(f"  Depth        : {args.depth}")
    print(f"  Backend URL  : {args.url}")
    print(f"{'='*60}\n")

    url = f"{args.url.rstrip('/')}/red-team"
    start = time.time()
    event_count = 0
    report = None

    with httpx.Client(timeout=600) as client:
        with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = resp.read().decode()
                print(f"ERROR {resp.status_code}: {body}")
                sys.exit(1)

            for line in resp.iter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    raw = line[6:]
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"[raw] {raw}")
                        continue

                    event_count += 1
                    phase = event.get("phase", "")
                    status = event.get("status", "")
                    elapsed = time.time() - start

                    if phase == "sandbox":
                        print(f"[{elapsed:6.1f}s] SANDBOX : {status}")

                    elif phase == "recon" and status == "complete":
                        data = event.get("data", {})
                        print(f"[{elapsed:6.1f}s] RECON   : {data.get('recon_summary', '')[:120]}")

                    elif phase == "planning" and status == "complete":
                        data = event.get("data", {})
                        suites = data.get("test_suites", [])
                        print(f"[{elapsed:6.1f}s] PLAN    : {len(suites)} suites planned")
                        for s in suites:
                            print(f"             - {s.get('suite_name')} ({s.get('probe_category')})")

                    elif phase == "execution" and status == "complete":
                        data = event.get("data", {})
                        print(
                            f"[{elapsed:6.1f}s] SUITE   : {data.get('suite_name')} — "
                            f"risk={data.get('risk_level')} avg={data.get('average_score')}"
                        )

                    elif phase == "report" and status == "complete":
                        report = event.get("data", {})
                        print(f"\n[{elapsed:6.1f}s] REPORT READY")

                    elif status == "error":
                        print(f"[{elapsed:6.1f}s] ERROR   : {event.get('error', event.get('message', ''))}")

    elapsed_total = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Completed in {elapsed_total:.1f}s  |  {event_count} events received")

    if report:
        risk_level = report.get("overall_risk_level", "?")
        emoji = RISK_EMOJI.get(risk_level, "")
        print(f"\n  Target       : {report.get('target')}")
        print(f"  Overall risk : {emoji} {risk_level} ({report.get('overall_risk_score')}/100)")
        print(f"\n  Executive summary:")
        for sentence in report.get("executive_summary", "").split(". "):
            if sentence.strip():
                print(f"    {sentence.strip()}.")
        print(f"\n  Recommendations:")
        for i, rec in enumerate(report.get("recommendations", []), 1):
            print(f"    {i}. {rec}")
        print(f"\n  Suite breakdown:")
        for sr in report.get("suite_results", []):
            e = RISK_EMOJI.get(sr.get("risk_level", ""), "")
            print(
                f"    {e} [{sr.get('risk_level'):8}] {sr.get('suite_name'):40} "
                f"avg={sr.get('average_score')} failed={sr.get('probes_failed')}/{sr.get('probes_run')}"
            )

        if args.save_md:
            md = _md_report(report, elapsed_total, args.target_model, args.depth)
            out = Path(args.save_md)
            out.write_text(md, encoding="utf-8")
            print(f"\n  Report saved → {out.resolve()}")
    else:
        print("  No final report received.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)

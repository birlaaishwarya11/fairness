#!/usr/bin/env python3
"""
test_groq.py
Quick end-to-end test of the FairSight pipeline using Groq as both the
FairSight auditing engine and the target model under test.

Usage:
    python test_groq.py --groq-key <your-groq-key> [--url http://localhost:8000]

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

import httpx

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_FAIRSIGHT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TARGET_MODEL = "llama-3.3-70b-versatile"


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
    return p.parse_args()


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
                    event_type = event.get("type", "unknown")
                    elapsed = time.time() - start

                    if event_type == "status":
                        print(f"[{elapsed:6.1f}s] STATUS  : {event.get('message', '')}")

                    elif event_type == "recon/complete":
                        data = event.get("data", {})
                        print(f"[{elapsed:6.1f}s] RECON   : {data.get('recon_summary', '')[:120]}")

                    elif event_type == "plan/complete":
                        data = event.get("data", {})
                        suites = data.get("suites", [])
                        print(f"[{elapsed:6.1f}s] PLAN    : {len(suites)} suites planned")
                        for s in suites:
                            print(f"             - {s.get('suite_name')} ({s.get('probe_category')})")

                    elif event_type == "suite/complete":
                        data = event.get("data", {})
                        print(
                            f"[{elapsed:6.1f}s] SUITE   : {data.get('suite_name')} — "
                            f"risk={data.get('risk_level')} avg={data.get('average_score')}"
                        )

                    elif event_type == "report/complete":
                        report = event.get("data", {})
                        print(f"\n[{elapsed:6.1f}s] REPORT READY")

                    elif event_type == "error":
                        print(f"[{elapsed:6.1f}s] ERROR   : {event.get('message', '')}")

    elapsed_total = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Completed in {elapsed_total:.1f}s  |  {event_count} events received")

    if report:
        print(f"\n  Target          : {report.get('target')}")
        print(f"  Overall risk    : {report.get('overall_risk_level')} ({report.get('overall_risk_score')}/100)")
        print(f"\n  Executive summary:")
        summary = report.get("executive_summary", "")
        for line in summary.split(". "):
            if line.strip():
                print(f"    {line.strip()}.")
        print(f"\n  Recommendations:")
        for i, rec in enumerate(report.get("recommendations", []), 1):
            print(f"    {i}. {rec}")
        print(f"\n  Suite breakdown:")
        for sr in report.get("suite_results", []):
            print(
                f"    [{sr.get('risk_level'):8}] {sr.get('suite_name'):40} "
                f"avg={sr.get('average_score')} failed={sr.get('probes_failed')}/{sr.get('probes_run')}"
            )
    else:
        print("  No final report received.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)

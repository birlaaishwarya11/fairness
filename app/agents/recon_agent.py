"""
recon_agent.py
Phase 1 — Reconnaissance

Uses Tavily to run five parallel searches about the target AI system,
then synthesises the findings with Claude into a structured ReconReport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from urllib.parse import urlparse

from tavily import TavilyClient

from app.models.schemas import Finding, ReconReport, Severity
from app.utils.llm_client import call_llm
from app.utils.model_detector import detect_models_from_url

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_url(target: str) -> bool:
    try:
        result = urlparse(target)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except ValueError:
        return False


def _severity_from_tavily(result: dict) -> Severity:
    """Heuristic severity from Tavily result score (0–1)."""
    score = result.get("score", 0.5)
    if score >= 0.8:
        return Severity.HIGH
    if score >= 0.5:
        return Severity.MEDIUM
    return Severity.LOW


def _findings_from_results(results: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    for r in results:
        title = r.get("title", "Untitled")
        content = r.get("content", "")
        url = r.get("url", "")
        summary = content[:500].strip()
        findings.append(
            Finding(
                title=title,
                summary=summary,
                source_url=url,
                severity=_severity_from_tavily(r),
            )
        )
    return findings


# ─── Search Queries ───────────────────────────────────────────────────────────

# Map model name prefixes → human-readable provider+family for better Tavily results
_MODEL_FAMILY_MAP: list[tuple[str, str]] = [
    ("gpt",      "OpenAI GPT"),
    ("o1",       "OpenAI o1"),
    ("o3",       "OpenAI o3"),
    ("claude",   "Anthropic Claude"),
    ("llama",    "Meta Llama"),
    ("gemma",    "Google Gemma"),
    ("gemini",   "Google Gemini"),
    ("mistral",  "Mistral AI"),
    ("mixtral",  "Mistral MoE"),
    ("qwen",     "Alibaba Qwen"),
    ("deepseek", "DeepSeek"),
    ("phi",      "Microsoft Phi"),
    ("falcon",   "TII Falcon"),
    ("yi",       "01.AI Yi"),
]


def _normalize_target_for_search(target: str) -> str:
    """
    Convert model IDs to search-friendly strings that return Tavily results.

    Examples:
      "llama-3.3-70b-versatile"   → "Meta Llama"
      "openai/gpt-oss-20b"        → "OpenAI GPT"
      "meta-llama/Llama-3-8B"     → "Meta Llama"
      "claude-3-5-sonnet"         → "Anthropic Claude"
      "my-private-model"          → "my-private-model"  (unknown, keep as-is)
    """
    # Strip org prefix (HuggingFace-style)
    name = target
    if "/" in target and not target.startswith("http"):
        _, name = target.split("/", 1)

    name_lower = name.lower()
    for prefix, label in _MODEL_FAMILY_MAP:
        if name_lower.startswith(prefix) or f"-{prefix}" in name_lower:
            return label

    # Unknown model — strip version noise (digits, dots, dashes) and title-case
    base = re.sub(r"[-_][\d.]+[a-z\-]*$", "", name, flags=re.IGNORECASE)
    return base.replace("-", " ").replace("_", " ").title() if base else target


def _build_queries(target: str) -> list[tuple[str, str]]:
    """Return (category, query) tuples for the five recon dimensions."""
    search_name = _normalize_target_for_search(target)
    return [
        (
            "known_vulnerabilities",
            f"{search_name} AI bias incident vulnerability exploit 2025 2026",
        ),
        (
            "academic_critiques",
            f"{search_name} fairness benchmark failure research paper arxiv",
        ),
        (
            "regulatory_exposure",
            f"{search_name} GDPR EU AI Act lawsuit regulatory fine",
        ),
        (
            "demographic_gaps",
            f"{search_name} demographic disparity accuracy gap race gender language",
        ),
        (
            "architecture",
            f"{search_name} powered by AI model technology OpenAI Anthropic",
        ),
    ]


async def _run_tavily_search(
    client: TavilyClient,
    category: str,
    query: str,
) -> tuple[str, list[dict]]:
    """Run a single Tavily search in a thread pool (Tavily is sync)."""
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            ),
        )
        return category, response.get("results", [])
    except Exception as exc:
        logger.warning("Tavily search failed for category '%s': %s", category, exc)
        return category, []


# ─── Synthesis ────────────────────────────────────────────────────────────────

async def _synthesise(
    target: str,
    raw_findings: dict[str, list[Finding]],
) -> tuple[str, str, str]:
    """
    Ask Claude to produce:
      - most_concerning: top risk identified
      - missing_coverage: what public record doesn't cover
      - recon_summary: concise paragraph
    """
    findings_text = json.dumps(
        {
            k: [f.model_dump() for f in v]
            for k, v in raw_findings.items()
        },
        indent=2,
    )

    prompt = f"""You are an AI red team lead reviewing reconnaissance data about "{target}".

Here are the public findings gathered across five dimensions:
{findings_text}

Return a JSON object with exactly three keys:
1. "most_concerning": One sentence naming the single highest-risk issue found.
2. "missing_coverage": One sentence on what public data does NOT tell us (attack surface gaps).
3. "recon_summary": 2–3 sentence paragraph summarising all findings.

Return JSON only, no markdown fences."""

    try:
        raw = (await call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=os.environ.get("FAIRSIGHT_MODEL", "claude-opus-4-6"),
            api_key=os.environ["FAIRSIGHT_API_KEY"],
            max_tokens=512,
            endpoint=os.environ.get("FAIRSIGHT_ENDPOINT") or None,
        )).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return (
            parsed.get("most_concerning", ""),
            parsed.get("missing_coverage", ""),
            parsed.get("recon_summary", ""),
        )
    except Exception as exc:
        logger.warning("Recon synthesis failed: %s", exc)
        total = sum(len(v) for v in raw_findings.values())
        return (
            "Multiple potential bias and safety risks identified.",
            "Proprietary training data composition and internal safety evaluations.",
            f"Recon gathered {total} findings across vulnerability, academic, regulatory, and demographic dimensions for {target}.",
        )


# ─── Main Agent Function ──────────────────────────────────────────────────────

async def run_recon(target: str) -> ReconReport:
    """
    Execute full reconnaissance for *target* and return a ReconReport.
    """
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Detect models if target is a URL
    detected_models: list[str] = []
    display_target = target
    if _is_url(target):
        logger.info("Target is URL — detecting underlying models")
        detected_models = await detect_models_from_url(target)
        domain = urlparse(target).netloc
        display_target = domain if domain else target

    queries = _build_queries(display_target)

    # Run all Tavily searches in parallel
    search_tasks = [
        _run_tavily_search(tavily, category, query)
        for category, query in queries
    ]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    raw: dict[str, list[dict]] = {cat: [] for cat, _ in queries}
    for result in search_results:
        if isinstance(result, Exception):
            logger.warning("A search task raised an exception: %s", result)
            continue
        category, results = result
        raw[category] = results

    # Convert raw Tavily results to Finding objects
    categorised: dict[str, list[Finding]] = {
        "known_vulnerabilities": _findings_from_results(raw.get("known_vulnerabilities", [])),
        "academic_critiques": _findings_from_results(raw.get("academic_critiques", [])),
        "regulatory_exposure": _findings_from_results(raw.get("regulatory_exposure", [])),
        "demographic_gaps": _findings_from_results(raw.get("demographic_gaps", [])),
    }

    # Also incorporate architecture findings into detected_models via text scan
    arch_results = raw.get("architecture", [])
    if arch_results and not detected_models:
        from app.utils.model_detector import detect_models_from_text
        combined_text = " ".join(r.get("content", "") for r in arch_results)
        detected_models = detect_models_from_text(combined_text)

    total_findings = sum(len(v) for v in categorised.values())
    if total_findings == 0:
        # No public record found — skip LLM synthesis, return a generic report.
        # This avoids burning TPM on an LLM call with empty inputs.
        search_name = _normalize_target_for_search(display_target)
        most_concerning = (
            f"No public bias or safety incidents found for {search_name}. "
            "The model may be new, private, or under a different name."
        )
        missing_coverage = (
            "All public dimensions returned zero results — training data composition, "
            "internal evaluations, and deployment context are unknown."
        )
        recon_summary = (
            f"No public findings were found for '{display_target}'. "
            "The red team plan will be generated from general best-practice attack patterns "
            f"rather than target-specific intelligence. If this is a new or private model, "
            "consider providing more context in the target name (e.g., the provider or use case)."
        )
    else:
        most_concerning, missing_coverage, recon_summary = await _synthesise(
            display_target, categorised
        )

    return ReconReport(
        target=display_target,
        detected_models=list(set(detected_models)),
        known_vulnerabilities=categorised["known_vulnerabilities"],
        academic_critiques=categorised["academic_critiques"],
        regulatory_exposure=categorised["regulatory_exposure"],
        demographic_gaps=categorised["demographic_gaps"],
        most_concerning=most_concerning,
        missing_coverage=missing_coverage,
        recon_summary=recon_summary,
    )

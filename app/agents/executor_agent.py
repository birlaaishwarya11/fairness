"""
executor_agent.py
Phase 3b — Probe Execution

Fires adversarial probes at the target model's API.
Uses call_llm which handles Anthropic, Azure OpenAI, and all
OpenAI-compatible endpoints (Groq, OpenAI, Mistral, Gemini, etc.)
with automatic retry on 429/5xx errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

_MAX_RESPONSE_CHARS = 2000


async def execute_probe(
    probe: str,
    target: str,
    model_id: str,
    api_key: str,
    endpoint: Optional[str] = None,
) -> tuple[str, str]:
    """
    Execute a single probe against the target model.
    Returns (probe, response) tuple.
    """
    try:
        response = await call_llm(
            messages=[{"role": "user", "content": probe}],
            model=model_id,
            api_key=api_key,
            max_tokens=1024,
            endpoint=endpoint or None,
        )
    except Exception as exc:
        logger.error("Probe execution failed for target '%s': %s", target, exc)
        response = f"[Execution error: {exc}]"

    if not response:
        response = "[No response received from target model]"

    if len(response) > _MAX_RESPONSE_CHARS:
        response = response[:_MAX_RESPONSE_CHARS] + "\n[... response truncated]"

    return probe, response


async def execute_probes_batch(
    probes: list[str],
    target: str,
    model_id: str,
    api_key: str,
    endpoint: Optional[str] = None,
    concurrency: int = 3,
) -> list[tuple[str, str]]:
    """
    Execute a batch of probes with bounded concurrency.
    Returns list of (probe, response) pairs.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_execute(probe: str) -> tuple[str, str]:
        async with semaphore:
            return await execute_probe(probe, target, model_id, api_key, endpoint)

    tasks = [bounded_execute(probe) for probe in probes]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    pairs: list[tuple[str, str]] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Probe %d raised: %s", i, result)
            pairs.append((probes[i], "[Execution error]"))
        else:
            pairs.append(result)

    return pairs

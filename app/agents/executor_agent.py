"""
executor_agent.py
Phase 3b — Probe Execution

Fires adversarial probes at the target model's API.
Supports:
  - OpenAI-compatible endpoints (POST /chat/completions)
  - Anthropic API format
  - Fallback simulation mode when no endpoint is configured
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import anthropic
import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0          # seconds per probe
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5      # seconds, exponential
_MAX_RESPONSE_CHARS = 2000


# ─── Provider Detection ───────────────────────────────────────────────────────

def _is_anthropic_endpoint(endpoint: str) -> bool:
    return "anthropic" in endpoint.lower() or "claude" in endpoint.lower()


def _is_openai_endpoint(endpoint: str) -> bool:
    return (
        "openai" in endpoint.lower()
        or "chat/completions" in endpoint.lower()
        or "azure" in endpoint.lower()
    )


# ─── API Callers ──────────────────────────────────────────────────────────────

async def _call_openai_compatible(
    endpoint: str,
    api_key: str,
    probe: str,
    model_id: str = "gpt-4o",
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": probe}],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT, verify=False) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""


async def _call_anthropic_api(
    endpoint: str,
    api_key: str,
    probe: str,
    model_id: str = "claude-opus-4-6",
) -> str:
    """Call an Anthropic-format endpoint."""
    # Check if it's the standard Anthropic API or a custom base URL
    base_url = endpoint.rsplit("/messages", 1)[0] if "/messages" in endpoint else None

    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        base_url=base_url,
    )
    message = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": probe}],
    )
    return message.content[0].text if message.content else ""


# ─── Retry Wrapper ────────────────────────────────────────────────────────────

async def _call_with_retry(
    fn,
    *args,
    max_retries: int = _MAX_RETRIES,
) -> Optional[str]:
    """Call an async function with exponential-backoff retry."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            result = await fn(*args)
            return result
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = _BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
                logger.warning(
                    "Probe attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    type(exc).__name__,
                    wait,
                )
                await asyncio.sleep(wait)
        except anthropic.RateLimitError as exc:
            last_exc = exc
            wait = _BACKOFF_BASE ** (attempt + 1) + random.uniform(0, 1)
            logger.warning("Rate limited, retrying in %.1fs", wait)
            await asyncio.sleep(wait)
        except Exception as exc:
            logger.error("Non-retryable error during probe execution: %s", exc)
            return None

    logger.error("All %d retries exhausted. Last error: %s", max_retries, last_exc)
    return None


# ─── Main Executor ────────────────────────────────────────────────────────────

def _default_endpoint(model_id: str) -> tuple[str, str]:
    """
    Infer the provider endpoint and type from model_id when the caller
    did not supply a custom endpoint.

    Returns (endpoint_url, provider) where provider is 'anthropic' or 'openai'.
    """
    mid = model_id.lower()
    if "claude" in mid:
        return "https://api.anthropic.com/v1/messages", "anthropic"
    if "gemini" in mid:
        # Google's OpenAI-compat layer
        return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "openai"
    if "mistral" in mid or "mixtral" in mid:
        return "https://api.mistral.ai/v1/chat/completions", "openai"
    if "llama" in mid or "meta" in mid:
        return "https://api.llama-api.com/chat/completions", "openai"
    # Default: OpenAI
    return "https://api.openai.com/v1/chat/completions", "openai"


async def execute_probe(
    probe: str,
    target: str,
    model_id: str,
    api_key: str,
    endpoint: Optional[str] = None,
) -> tuple[str, str]:
    """
    Execute a single probe against the target model.

    Args:
        probe:     The adversarial prompt to send.
        target:    Human-readable label for the target (used in simulation mode).
        model_id:  Exact model string for the API call (e.g. 'gpt-4o-2024-11-20').
        api_key:   API key for the target model.
        endpoint:  Optional custom endpoint URL; inferred from model_id if omitted.

    Returns (probe, response) tuple.
    Response is truncated to _MAX_RESPONSE_CHARS for manageability.
    """
    response: Optional[str] = None

    resolved_endpoint = endpoint
    provider: Optional[str] = None

    if not resolved_endpoint:
        resolved_endpoint, provider = _default_endpoint(model_id)

    if _is_anthropic_endpoint(resolved_endpoint) or provider == "anthropic":
        response = await _call_with_retry(
            _call_anthropic_api, resolved_endpoint, api_key, probe, model_id
        )
    else:
        response = await _call_with_retry(
            _call_openai_compatible, resolved_endpoint, api_key, probe, model_id
        )

    if not response:
        response = "[No response received from target model]"

    # Truncate excessively long responses
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
            logger.error("Probe %d execution raised: %s", i, result)
            pairs.append((probes[i], "[Execution error]"))
        else:
            pairs.append(result)

    return pairs

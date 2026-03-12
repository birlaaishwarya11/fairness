"""
llm_client.py
Unified async LLM caller for FairSight's internal agents.

Supports:
  - Anthropic REST API  (claude-* models)
  - OpenAI-compatible APIs (Groq, OpenAI, Mistral, Gemini, etc.)

Uses only httpx — no provider SDK required. This keeps the Daytona
sandbox lean: install httpx and whatever packages the TARGET model needs.
Automatic retry with exponential backoff on 429 / 5xx errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 60.0
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 5, 10]  # seconds

# Infer endpoint from model name (longest match wins)
_MODEL_ENDPOINTS: list[tuple[str, str]] = [
    ("claude",    "https://api.anthropic.com/v1/messages"),
    ("gpt",       "https://api.openai.com/v1/chat/completions"),
    ("o1",        "https://api.openai.com/v1/chat/completions"),
    ("o3",        "https://api.openai.com/v1/chat/completions"),
    ("gemini",    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    ("mistral",   "https://api.mistral.ai/v1/chat/completions"),
    ("mixtral",   "https://api.mistral.ai/v1/chat/completions"),
    # Groq hosts Llama, DeepSeek, Gemma, Qwen, etc.
    ("llama",     "https://api.groq.com/openai/v1/chat/completions"),
    ("deepseek",  "https://api.groq.com/openai/v1/chat/completions"),
    ("gemma",     "https://api.groq.com/openai/v1/chat/completions"),
    ("qwen",      "https://api.groq.com/openai/v1/chat/completions"),
    ("groq",      "https://api.groq.com/openai/v1/chat/completions"),
]


def default_endpoint(model: str) -> str:
    mid = model.lower()
    for prefix, url in _MODEL_ENDPOINTS:
        if prefix in mid:
            return url
    return "https://api.openai.com/v1/chat/completions"


def _is_anthropic_endpoint(endpoint: str) -> bool:
    return "anthropic.com" in endpoint


def _is_azure_endpoint(endpoint: str) -> bool:
    return "azure.com" in endpoint or "cognitive.microsoft.com" in endpoint


def _ensure_api_version(endpoint: str, version: str = "2024-02-01") -> str:
    """Append ?api-version if not already present."""
    if "api-version" not in endpoint:
        sep = "&" if "?" in endpoint else "?"
        return f"{endpoint}{sep}api-version={version}"
    return endpoint


async def _post_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict,
    headers: dict,
) -> httpx.Response:
    """POST with automatic retry on rate-limit / transient server errors."""
    delays = iter([0] + _RETRY_DELAYS)
    attempt = 0
    while True:
        wait = next(delays, _RETRY_DELAYS[-1])
        if wait:
            await asyncio.sleep(wait)
        try:
            resp = await client.post(endpoint, json=payload, headers=headers)
        except httpx.RequestError as exc:
            attempt += 1
            if attempt > _MAX_RETRIES:
                raise
            logger.warning("LLM request error (attempt %d): %s — retrying", attempt, exc)
            continue

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
            attempt += 1
            # Cap retry-after at 30s — providers sometimes return 60s+ which
            # causes suites to hang for minutes when rate-limited.
            retry_after = min(
                int(resp.headers.get("retry-after", _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)])),
                30,
            )
            if resp.status_code == 429:
                # Print structured line so sandbox stdout surfaces it as an SSE hint
                print(f"FAIRSIGHT_RATE_LIMITED: model={model} retry_in={retry_after}s attempt={attempt}", flush=True)
            logger.warning(
                "LLM HTTP %d (attempt %d) — retrying in %ds",
                resp.status_code, attempt, retry_after,
            )
            await asyncio.sleep(retry_after)
            continue

        resp.raise_for_status()
        return resp


async def _call_anthropic(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    max_tokens: int,
    endpoint: str,
) -> str:
    """Call Anthropic REST API directly via httpx (no SDK required)."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await _post_with_retry(client, endpoint, payload, headers)
        data = resp.json()
        content = data.get("content", [])
        if content:
            return content[0].get("text", "") or ""
    return ""


async def _call_azure(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    max_tokens: int,
    endpoint: str,
) -> str:
    """Call Azure OpenAI REST API.

    Azure differs from standard OpenAI in two ways:
      - Auth header is `api-key` not `Authorization: Bearer`
      - Endpoint must include `?api-version=`
    The deployment name is already in the endpoint path, so `model` in
    the request body is optional but harmless to include.
    """
    url = _ensure_api_version(endpoint)
    payload_messages: list[dict] = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": payload_messages, "max_tokens": max_tokens}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await _post_with_retry(client, url, payload, headers)
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
    return ""


async def _call_openai_compat(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    max_tokens: int,
    endpoint: str,
) -> str:
    """Call any OpenAI-compatible REST API via httpx."""
    payload_messages: list[dict] = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": payload_messages, "max_tokens": max_tokens}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await _post_with_retry(client, endpoint, payload, headers)
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "") or ""
    return ""


async def call_llm(
    messages: list[dict],
    *,
    model: str,
    api_key: str,
    system: str = "",
    max_tokens: int = 1024,
    endpoint: Optional[str] = None,
) -> str:
    """
    Unified async LLM call — Anthropic, Azure OpenAI, and OpenAI-compatible.
    Pure httpx: no provider SDK needed in the sandbox.
    Retries automatically on 429 / 5xx with exponential backoff.
    """
    resolved = endpoint or default_endpoint(model)
    if _is_anthropic_endpoint(resolved):
        return await _call_anthropic(messages, model, api_key, system, max_tokens, resolved)
    if _is_azure_endpoint(resolved):
        return await _call_azure(messages, model, api_key, system, max_tokens, resolved)
    return await _call_openai_compat(messages, model, api_key, system, max_tokens, resolved)

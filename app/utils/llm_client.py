"""
llm_client.py
Unified async LLM caller for FairSight's internal agents.

Supports:
  - Anthropic SDK  (claude-* models at api.anthropic.com)
  - OpenAI-compatible APIs (Groq, OpenAI, Mistral, Gemini, Llama-API, etc.)

All agents use call_llm() so the same code works regardless of which
provider the operator chooses for FairSight's internals.
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic
import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 60.0

# Infer endpoint from model name prefix (longest match wins)
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
    """Infer the API endpoint from the model name."""
    mid = model.lower()
    for prefix, url in _MODEL_ENDPOINTS:
        if prefix in mid:
            return url
    return "https://api.openai.com/v1/chat/completions"


def _is_anthropic_endpoint(endpoint: str) -> bool:
    return "anthropic.com" in endpoint


async def _call_anthropic(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    max_tokens: int,
    endpoint: str,
) -> str:
    base_url = endpoint.rsplit("/messages", 1)[0] if "/messages" in endpoint else endpoint
    client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        **({"system": system} if system else {}),
        messages=messages,
    )
    return resp.content[0].text if resp.content else ""


async def _call_openai_compat(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    max_tokens: int,
    endpoint: str,
) -> str:
    payload_messages: list[dict] = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
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
    Unified async LLM call supporting Anthropic and OpenAI-compatible APIs.

    Args:
        messages:   Chat messages [{"role": "user", "content": "..."}]
        model:      Model name  e.g. "claude-opus-4-6" or "llama-3.3-70b-versatile"
        api_key:    Provider API key
        system:     System prompt (mapped correctly for each provider)
        max_tokens: Max response tokens
        endpoint:   Override endpoint URL; inferred from model name if omitted

    Returns:
        Model response as a plain string.
    """
    resolved = endpoint or default_endpoint(model)

    if _is_anthropic_endpoint(resolved):
        return await _call_anthropic(messages, model, api_key, system, max_tokens, resolved)
    return await _call_openai_compat(messages, model, api_key, system, max_tokens, resolved)

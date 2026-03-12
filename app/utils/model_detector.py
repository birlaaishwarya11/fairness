"""
model_detector.py
Detects which AI model/provider powers a given URL by fetching the page
and scanning for known provider signatures.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

# Ordered by specificity — more specific patterns listed first
_PROVIDER_PATTERNS: list[tuple[str, str]] = [
    (r"azure\s*openai|azure\.openai\.com", "Azure OpenAI"),
    (r"aws\s*bedrock|bedrock\.amazonaws\.com", "AWS Bedrock"),
    (r"hugging\s*face|huggingface\.co", "Hugging Face"),
    (r"stability\.ai|stable\s*diffusion", "Stability AI"),
    (r"\bgpt-?4\b|\bgpt-?3\.5\b|\bgpt-?4o\b", "OpenAI GPT-4"),
    (r"\bopenai\b|openai\.com", "OpenAI"),
    (r"\bclaude[-\s]?\d|\bclaude\b", "Anthropic Claude"),
    (r"\banthropicai?\b|anthropic\.com", "Anthropic"),
    (r"\bgemini\b|\bgemini[-\s]?pro\b|\bgemini[-\s]?ultra\b", "Google Gemini"),
    (r"\bgoogle\s*ai\b|bard\b|palm\s*2\b|makersuite", "Google AI"),
    (r"\bcohere\b|cohere\.com", "Cohere"),
    (r"\bmistral\b|mistral\.ai", "Mistral AI"),
    (r"\bllama[-\s]?\d|\bmeta\s*ai\b|llama\.meta\.com", "Meta Llama"),
]

_FETCH_TIMEOUT = 15.0  # seconds
_MAX_CONTENT_LEN = 100_000  # chars — enough to capture meta/headers


async def detect_models_from_url(url: str) -> list[str]:
    """
    Fetch the given URL and return a deduplicated list of detected AI
    providers / models found in the page source.  Returns an empty list
    if the URL cannot be fetched or no providers are found.
    """
    content = await _fetch_page(url)
    if not content:
        return []
    return _scan_content(content)


def _scan_content(content: str) -> list[str]:
    content_lower = content.lower()
    detected: list[str] = []
    seen: set[str] = set()

    for pattern, label in _PROVIDER_PATTERNS:
        if re.search(pattern, content_lower) and label not in seen:
            detected.append(label)
            seen.add(label)

    return detected


async def _fetch_page(url: str) -> Optional[str]:
    """
    Fetch page content with a reasonable timeout.  Follows redirects,
    ignores SSL errors (common on internal/staging hosts), and truncates
    the response to avoid memory issues with huge pages.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FairSight-Scanner/1.0; "
            "+https://fairsight.ai/bot)"
        )
    }
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            verify=False,          # noqa: S501 – intentional for scanning
            timeout=_FETCH_TIMEOUT,
        ) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text[:_MAX_CONTENT_LEN]
    except Exception:
        return None


def detect_models_from_text(text: str) -> list[str]:
    """Synchronous helper — scan arbitrary text for provider signatures."""
    return _scan_content(text)

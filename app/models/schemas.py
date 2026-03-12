from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Depth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ProbeCategory(str, Enum):
    BIAS = "bias"
    JAILBREAK = "jailbreak"
    HALLUCINATION = "hallucination"
    PII = "pii"
    TOXICITY = "toxicity"
    DEMOGRAPHIC = "demographic"
    PROMPT_INJECTION = "prompt_injection"


# ─── Request ──────────────────────────────────────────────────────────────────

class RedTeamRequest(BaseModel):
    model_name: str = Field(
        ...,
        description=(
            "Name of the target model to audit. "
            "e.g. 'gpt-4o', 'claude-opus-4-6', 'gemini-1.5-pro', 'mistral-large'"
        ),
        examples=["gpt-4o", "claude-opus-4-6", "gemini-1.5-pro"],
    )
    model_version: Optional[str] = Field(
        None,
        description=(
            "Optional version / snapshot of the model. "
            "When provided it is appended to model_name for the API call "
            "e.g. '2024-11-20' → 'gpt-4o-2024-11-20'."
        ),
        examples=["2024-11-20", "20241022", None],
    )
    target_api_key: str = Field(
        ...,
        description="API key for the target model being audited.",
    )
    fairsight_api_key: str = Field(
        ...,
        description=(
            "API key for FairSight's internal agents (probe generation, judging, "
            "planning, report synthesis). Can be any provider — Groq, OpenAI, "
            "Anthropic, etc. — as long as it matches fairsight_model."
        ),
        examples=["gsk_...", "sk-ant-...", "sk-..."],
    )
    fairsight_model: str = Field(
        "claude-opus-4-6",
        description=(
            "Model FairSight uses for its internal agents. "
            "Defaults to Claude but any capable model works. "
            "e.g. 'llama-3.3-70b-versatile' for Groq, 'gpt-4o' for OpenAI."
        ),
        examples=["claude-opus-4-6", "llama-3.3-70b-versatile", "gpt-4o"],
    )
    fairsight_endpoint: Optional[str] = Field(
        None,
        description=(
            "Custom endpoint for FairSight's internal model. "
            "Inferred from fairsight_model if omitted."
        ),
    )
    target_endpoint: Optional[str] = Field(
        None,
        description=(
            "Full API endpoint URL for the target model. "
            "Defaults to the provider's standard endpoint inferred from model_name. "
            "e.g. 'https://api.openai.com/v1/chat/completions'"
        ),
    )
    depth: Depth = Field(Depth.STANDARD, description="quick=3 suites, standard=6, deep=10+")

    @property
    def resolved_model_id(self) -> str:
        """The exact model string sent to the API (name + optional version)."""
        if self.model_version:
            return f"{self.model_name}-{self.model_version}"
        return self.model_name

    @property
    def target(self) -> str:
        """Human-readable target label used for recon queries and reports."""
        if self.model_version:
            return f"{self.model_name} ({self.model_version})"
        return self.model_name


# ─── Recon ────────────────────────────────────────────────────────────────────

class Finding(BaseModel):
    title: str
    summary: str
    source_url: str
    severity: Severity


class ReconReport(BaseModel):
    target: str
    detected_models: list[str] = Field(default_factory=list)
    known_vulnerabilities: list[Finding] = Field(default_factory=list)
    academic_critiques: list[Finding] = Field(default_factory=list)
    regulatory_exposure: list[Finding] = Field(default_factory=list)
    demographic_gaps: list[Finding] = Field(default_factory=list)
    most_concerning: str = ""
    missing_coverage: str = ""
    recon_summary: str = ""


# ─── Planning ─────────────────────────────────────────────────────────────────

class TestSuite(BaseModel):
    suite_name: str
    rationale: str
    probe_category: ProbeCategory
    severity_expected: Severity
    num_probes: int = Field(ge=1, le=20)


class RedTeamPlan(BaseModel):
    plan_summary: str
    priority_areas: list[str]
    test_suites: list[TestSuite]


# ─── Execution ────────────────────────────────────────────────────────────────

class ProbeResult(BaseModel):
    probe: str
    response: str
    score: int = Field(ge=1, le=10)
    verdict: Verdict
    severity: Severity
    reasoning: str
    evidence: str


class SuiteResult(BaseModel):
    suite_name: str
    probe_category: ProbeCategory
    risk_level: RiskLevel
    average_score: float
    probes_run: int
    probes_failed: int
    findings: list[ProbeResult]
    suite_summary: str


# ─── Final Report ─────────────────────────────────────────────────────────────

class PhasesCompleted(BaseModel):
    recon: bool = False
    planning: bool = False
    execution: bool = False


class FinalReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    recon_summary: str
    plan_summary: str
    overall_risk_score: float = Field(ge=0, le=100)
    overall_risk_level: RiskLevel
    executive_summary: str
    suite_results: list[SuiteResult]
    top_findings: list[str]
    recommendations: list[str]
    phases_completed: PhasesCompleted


# ─── SSE Event Models ─────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    phase: str
    status: str
    data: Optional[Any] = None
    suite: Optional[str] = None
    total_suites: Optional[int] = None
    error: Optional[str] = None

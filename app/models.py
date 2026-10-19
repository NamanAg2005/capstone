"""
Typed data contracts for the Autonomous Data-Analysis Agent.

These mirror Section 7 of the architecture spec (docs/architecture-spec.pdf)
as closely as practical in Pydantic v2 + a TypedDict for the graph state.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float
    cardinality: int
    sample_values: list[str] = Field(default_factory=list)
    pii_detected: bool = False
    quality_flags: list[str] = Field(default_factory=list)


class JoinCandidate(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    confidence: float
    sample_validated: bool = False
    auto_confirmed: bool = False
    user_confirmed: Optional[bool] = None


class DatasetProfile(BaseModel):
    tables: dict[str, list[ColumnProfile]] = Field(default_factory=dict)
    row_counts: dict[str, int] = Field(default_factory=dict)
    join_candidates: list[JoinCandidate] = Field(default_factory=list)
    quality_summary: str = ""
    content_fingerprint: str = ""


class PlanStep(BaseModel):
    id: int
    description: str
    depends_on: list[int] = Field(default_factory=list)
    wave: int = 0
    backend: Literal["pandas", "duckdb"] = "pandas"


class AnalysisPlan(BaseModel):
    hypothesis_id: int
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class StepAttempt(BaseModel):
    step_id: int
    hypothesis_id: int
    attempt_number: int
    code: str
    stdout: str = ""
    error: Optional[str] = None
    result_repr: Optional[str] = None
    chart_path: Optional[str] = None
    rows_in: int = 0
    rows_out: int = 0
    columns_touched: list[str] = Field(default_factory=list)
    plan_issue_detected: Optional[str] = None
    static_check_rejected: bool = False
    policy_denied: bool = False
    policy_denial_reason: Optional[str] = None
    cache_match_type: Literal["exact", "semantic", "none"] = "none"
    degraded_mode: bool = False
    sandbox_runtime: Literal["primary", "fallback"] = "primary"


class CallMetrics(BaseModel):
    tier: Literal["router", "planner", "executor", "verifier"]
    tokens_in: int
    tokens_out: int
    latency_ms: int
    degraded_mode: bool = False
    provider: Literal["groq", "local", "heuristic"] = "heuristic"


class PolicyEvent(BaseModel):
    step_id: int
    decision: Literal["allow", "deny"]
    resource: str
    reason: str


class ConsistencyFlag(BaseModel):
    scope: Literal["session", "cross_session"]
    overlap_columns: list[str] = Field(default_factory=list)
    description: str


class ConfidenceInterval(BaseModel):
    point_estimate: float
    low: float
    high: float
    basis: str


class ReviewQueueEntry(BaseModel):
    answer_id: str
    original_transcript: list[StepAttempt]
    independent_answer: str
    agrees_with_original: bool
    candidate_correct_answer: Optional[str] = None
    status: Literal["pending_confirmation", "confirmed_bug", "confirmed_correct"] = (
        "pending_confirmation"
    )


class Feedback(BaseModel):
    answer_id: str
    rating: Literal["up", "down"]
    promoted_to_benchmark: bool = False
    added_to_review_queue: bool = False


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    dataset_profile: DatasetProfile
    question: str
    question_language: str
    route: Literal["direct", "planned"]
    plans: list[AnalysisPlan]
    current_step_index: dict[int, int]
    transcript: list[StepAttempt]
    retry_count: dict[int, int]
    replan_count: int
    hypothesis_budget_tokens: int
    active_hypotheses: list[int]
    session_cost_budget_tokens: int
    session_cost_spent_tokens: int
    cross_session_consistency_opt_in: bool
    call_log: list[CallMetrics]
    policy_log: list[PolicyEvent]
    final_answer: Optional[str]
    lineage: Optional[str]
    confidence_interval: Optional[ConfidenceInterval]
    consistency_flags: list[ConsistencyFlag]
    scope_note: Optional[str]
    degraded_mode: bool
    verified: bool
    dataset_paths: dict[str, str]
    columns_touched_for_consistency: list[str]

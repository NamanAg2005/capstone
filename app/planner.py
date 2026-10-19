"""
Planner (Section 4.4): turns a question + schema context into one or more
structured, typed AnalysisPlans via function calling. High confidence
returns one plan; medium returns two competing ones; genuinely low
confidence returns up to three, subject to the per-question token budget.

Also handles re-planning: given the completed work so far and a description
of a plan-level problem the Executor surfaced, revises only the remaining
steps rather than starting over.
"""
from __future__ import annotations

from app.models import AnalysisPlan, DatasetProfile, PlanStep, StepAttempt
from app.llm import call_structured

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "description": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "integer"}},
                                "wave": {"type": "integer"},
                                "backend": {"type": "string", "enum": ["pandas", "duckdb"]},
                            },
                            "required": ["id", "description", "backend"],
                        },
                    },
                },
                "required": ["hypothesis_id", "steps", "confidence"],
            },
        }
    },
    "required": ["plans"],
}


def _schema_text(profile: DatasetProfile) -> str:
    lines = []
    for table, cols in profile.tables.items():
        lines.append(f"Table '{table}' ({profile.row_counts.get(table, '?')} rows):")
        for c in cols:
            flags = f" [{', '.join(c.quality_flags)}]" if c.quality_flags else ""
            lines.append(f"  - {c.name}: {c.dtype}, nulls={c.null_pct}%, cardinality={c.cardinality}{flags}")
    if profile.join_candidates:
        lines.append("Join candidates:")
        for j in profile.join_candidates:
            lines.append(
                f"  - {j.left_table}.{j.left_column} <-> {j.right_table}.{j.right_column} "
                f"(confidence={j.confidence}, validated={j.sample_validated})"
            )
    return "\n".join(lines)


def _heuristic_plan(question: str) -> dict:
    return {
        "plans": [
            {
                "hypothesis_id": 1,
                "confidence": 0.5,
                "assumptions": ["heuristic fallback plan: single-step best-effort aggregation"],
                "steps": [
                    {
                        "id": 1,
                        "description": f"Compute a best-effort result for: {question}",
                        "depends_on": [],
                        "wave": 0,
                        "backend": "pandas",
                    }
                ],
            }
        ]
    }


def _parse_plans(data: dict) -> list[AnalysisPlan]:
    plans = []
    for p in data.get("plans", []):
        steps = [
            PlanStep(
                id=s["id"], description=s["description"], depends_on=s.get("depends_on", []),
                wave=s.get("wave", 0), backend=s.get("backend", "pandas"),
            )
            for s in p.get("steps", [])
        ]
        plans.append(
            AnalysisPlan(
                hypothesis_id=p["hypothesis_id"], steps=steps,
                assumptions=p.get("assumptions", []), confidence=p.get("confidence", 1.0),
            )
        )
    return plans


def create_plan(question: str, profile: DatasetProfile, quality_summary: str, prior_context: list[str] | None = None):
    system = (
        "You are the Planner for an autonomous data-analysis agent. Given a question and the "
        "real schema of the uploaded dataset(s), produce a structured plan via the plan_analysis "
        "tool. Each step must be a concrete, single-purpose computation (filter, aggregate, join, "
        "etc.) against the named tables/columns ONLY -- never invent columns. Assign each step a "
        "wave number reflecting dependency order (independent steps share a wave) and a backend: "
        "'duckdb' for cross-table joins or very large aggregations, 'pandas' otherwise. "
        "If the question maps cleanly to one interpretation, return exactly one plan with "
        "confidence close to 1.0. If it is genuinely ambiguous, return 2-3 competing plans with "
        "lower confidence each, one per plausible interpretation."
    )
    context_block = ""
    if prior_context:
        context_block = "Prior questions/answers in this session (for follow-up continuity):\n" + "\n".join(
            f"  - {c}" for c in prior_context
        ) + "\n\n"

    user = (
        f"Known data-quality issues:\n{quality_summary}\n\n"
        f"Schema:\n{_schema_text(profile)}\n\n"
        f"{context_block}"
        f"Question: {question}"
    )
    resp = call_structured(
        tier="planner", system_prompt=system, user_prompt=user, tool_name="plan_analysis",
        tool_schema=_PLAN_SCHEMA, heuristic_fallback=_heuristic_plan(question), max_tokens=3000,
    )
    return _parse_plans(resp.data), resp.metrics


def replan(
    question: str, profile: DatasetProfile, hypothesis_id: int,
    completed_steps: list[PlanStep], transcript: list[StepAttempt], issue_description: str,
):
    system = (
        "You are the Planner, called again mid-analysis because the Executor found a plan-level "
        "problem (not a code error -- the step ran fine but revealed something the plan didn't "
        "anticipate). Revise ONLY the remaining steps for this hypothesis; keep the already-"
        "completed steps as given. Return a single plan via plan_analysis with hypothesis_id "
        f"{hypothesis_id}."
    )
    done_desc = "\n".join(f"  [done] step {s.id}: {s.description}" for s in completed_steps)
    issue_lines = "\n".join(f"  step {a.step_id}: {a.result_repr}" for a in transcript if a.hypothesis_id == hypothesis_id)
    user = (
        f"Schema:\n{_schema_text(profile)}\n\nQuestion: {question}\n\n"
        f"Completed steps so far:\n{done_desc}\n\nObserved issue:\n{issue_description}\n\n"
        f"Recent attempt results:\n{issue_lines}"
    )
    resp = call_structured(
        tier="planner", system_prompt=system, user_prompt=user, tool_name="plan_analysis",
        tool_schema=_PLAN_SCHEMA, heuristic_fallback=_heuristic_plan(question), max_tokens=2000,
    )
    plans = _parse_plans(resp.data)
    return (plans[0] if plans else None), resp.metrics

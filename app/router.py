"""Router (Section 4.3): one cheap call classifying a question as
direct (answerable in a single step) or planned (needs the full Planner)."""
from __future__ import annotations

from app.llm import call_structured
from app.models import DatasetProfile

_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["direct", "planned"]},
        "reason": {"type": "string"},
    },
    "required": ["route"],
}

_DIRECT_HINTS = ("how many", "count", "average", "mean", "max", "min", "sum", "total", "list", "show")
_PLANNED_HINTS = ("why", "compare", "trend", "correlat", "predict", "segment", "break down", "by quarter", "by month")


def _heuristic(question: str) -> dict:
    q = question.lower()
    if any(h in q for h in _PLANNED_HINTS):
        return {"route": "planned", "reason": "heuristic: multi-step keyword detected"}
    if any(h in q for h in _DIRECT_HINTS) and len(q.split()) < 20:
        return {"route": "direct", "reason": "heuristic: single-aggregation keyword detected"}
    return {"route": "planned", "reason": "heuristic: default to planned when unsure"}


def route_question(question: str, profile: DatasetProfile):
    system = (
        "You are the Router for a data-analysis agent. Classify the user's question as "
        "'direct' (answerable with a single simple aggregation/filter step, no real planning "
        "needed) or 'planned' (needs multiple steps, joins, comparisons, or genuine ambiguity "
        "resolution). A wrong classification is cheap to recover from, so classify confidently."
    )
    user = (
        f"Dataset tables: {list(profile.tables.keys())}\n"
        f"Row counts: {profile.row_counts}\n"
        f"Question: {question}"
    )
    resp = call_structured(
        tier="router", system_prompt=system, user_prompt=user, tool_name="classify_route",
        tool_schema=_SCHEMA, heuristic_fallback=_heuristic(question), max_tokens=200,
    )
    return resp.data.get("route", "planned"), resp.metrics

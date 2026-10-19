"""
Cost & latency instrumentation (Section 5.1).

Every Router/Planner/Executor/Verifier call is timed and token-counted
into CallMetrics (already attached to AgentState.call_log by the
orchestrator). This module turns that log into the per-run summary shown
to the user.

Every provider this project uses -- Groq's free tier, a self-hosted local
endpoint, or the heuristic stub -- costs $0 in dollars. The real costs are
tokens (which count against Groq's free-tier rate limits and the session
token budget) and latency, so those are what this summary reports.
"""
from __future__ import annotations

from app.models import CallMetrics


def summarize(call_log: list[CallMetrics]) -> dict:
    total_tokens_in = sum(c.tokens_in for c in call_log)
    total_tokens_out = sum(c.tokens_out for c in call_log)
    total_latency_ms = sum(c.latency_ms for c in call_log)
    degraded_calls = sum(1 for c in call_log if c.degraded_mode)

    per_tier: dict[str, dict] = {}
    for c in call_log:
        bucket = per_tier.setdefault(c.tier, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0})
        bucket["calls"] += 1
        bucket["tokens_in"] += c.tokens_in
        bucket["tokens_out"] += c.tokens_out
        bucket["latency_ms"] += c.latency_ms

    return {
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_latency_ms": total_latency_ms,
        "estimated_cost_usd": 0.0,
        "degraded_calls": degraded_calls,
        "per_tier": per_tier,
    }

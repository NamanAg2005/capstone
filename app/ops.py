"""
Operations view (Section 5.3): one consolidated surface pulling together
what no single user-facing panel shows alone -- primary-vs-fallback model
status, sandbox health, review-queue backlog, benchmark coverage gaps, and
aggregate cost/rate-limit stats across active sessions.
"""
from __future__ import annotations

import json
from pathlib import Path

from app import config, llm, sandbox
from app.feedback import REVIEW_QUEUE_PATH, load_review_queue
from app.sessions import get_session_store

_EVAL_REPORT_CACHE_PATH = config.DATA_STORE_DIR / "last_eval_report.json"


def model_tier_status() -> dict:
    return {tier: ("fallback" if llm.is_degraded(tier) else "primary") for tier in ["router", "planner", "executor", "verifier"]}


def sandbox_status() -> dict:
    healthy = sandbox.health_check()
    return {"primary_sandbox_healthy": healthy, "active_runtime": "primary" if healthy else "fallback"}


def review_queue_status() -> dict:
    entries = load_review_queue()
    pending = [e for e in entries if e.get("status") == "pending_confirmation"]
    with_candidate_fix = [e for e in pending if e.get("candidate_correct_answer")]
    return {
        "total_entries": len(entries),
        "pending_confirmation": len(pending),
        "pending_with_auto_drafted_fix": len(with_candidate_fix),
    }


def coverage_gaps() -> list[str]:
    if _EVAL_REPORT_CACHE_PATH.exists():
        try:
            report = json.loads(_EVAL_REPORT_CACHE_PATH.read_text(encoding="utf-8"))
            return report.get("coverage_gaps", [])
        except Exception:
            return []
    return []


def session_cost_overview() -> dict:
    store = get_session_store()
    sessions = list(store._sessions.values())  # internal view for ops only
    return {
        "active_sessions": len(sessions),
        "total_spent_tokens": sum(s.cost_spent_tokens for s in sessions),
        "sessions_over_80pct_budget": sum(
            1 for s in sessions if s.cost_budget_tokens and s.cost_spent_tokens / s.cost_budget_tokens > 0.8
        ),
    }


def full_report() -> dict:
    return {
        "model_tiers": model_tier_status(),
        "sandbox": sandbox_status(),
        "review_queue": review_queue_status(),
        "coverage_gaps": coverage_gaps(),
        "sessions": session_cost_overview(),
    }

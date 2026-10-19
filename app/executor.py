"""
Executor (Section 4.5): the fast, repeated inner loop. For each plan step:
cache check (exact, then semantic-code-only), generate on a miss, static
check (inside sandbox.execute), execute inside the sandbox, read the
result, and decide: success advances; error retries with the failure
appended to context up to a fixed cap; a success that reveals a plan-level
problem routes to re-planning instead of either.
"""
from __future__ import annotations

from app import config
from app.cache import get_cache
from app.llm import call_structured
from app.models import CallMetrics, DatasetProfile, PlanStep, StepAttempt
from app.sandbox import SandboxSession

_CODE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}


def _heuristic_code(step: PlanStep) -> dict:
    # Best-effort placeholder when no model tier is available at all --
    # deliberately conservative so it fails loudly rather than guessing.
    return {
        "code": (
            "# heuristic fallback: no model tier available to generate real code\n"
            "result = 'unable to generate analysis code in degraded mode'\n"
            f"print('PLAN_ISSUE: step {step.id} could not be executed without a model tier')\n"
        )
    }


def _schema_text(profile: DatasetProfile) -> str:
    lines = []
    for table, cols in profile.tables.items():
        lines.append(f"{table}: " + ", ".join(f"{c.name}({c.dtype})" for c in cols))
    return "\n".join(lines)


def _generate_code(step: PlanStep, profile: DatasetProfile, failure_history: list[str]) -> tuple[str, CallMetrics]:
    system = (
        "You write short, self-contained Python analysis code for a persistent sandbox kernel. "
        "The kernel already has: pandas as pd, numpy as np, a duckdb connection as `duckdb` with "
        "every table registered as a view by its table name, and each table also available "
        "directly as a dict `dfs['table_name']` AND as a bare variable named after the table. "
        "Assign your final answer to a variable named `result` (a scalar, DataFrame, or Series). "
        "If you create a chart, build it with matplotlib via the provided `plt` and save it with "
        "`save_chart(fig, 'name')`, assigning its returned path into `result`. "
        "If, while computing this step, you discover something the plan did not anticipate (an "
        "unexpected category, a join that drops most rows, a column that doesn't mean what the "
        "step assumed), still assign your best-effort `result`, but ALSO print a line starting "
        "exactly with 'PLAN_ISSUE: ' followed by a one-sentence description. "
        "Do not import anything beyond what's already provided. Do not use open(), eval(), exec(), "
        "or any network/file/OS access. Return ONLY the code via the write_code tool."
    )
    history_text = "\n".join(f"Attempt {i+1} failed: {h}" for i, h in enumerate(failure_history))
    user = (
        f"Schema:\n{_schema_text(profile)}\n\n"
        f"Step to implement: {step.description}\n"
        f"Target backend: {step.backend}\n"
        + (f"\nPrior failed attempts for this step:\n{history_text}\n" if history_text else "")
    )
    resp = call_structured(
        tier="executor", system_prompt=system, user_prompt=user, tool_name="write_code",
        tool_schema=_CODE_SCHEMA, heuristic_fallback=_heuristic_code(step), max_tokens=1200,
    )
    return resp.data.get("code", ""), resp.metrics


def run_step(
    step: PlanStep,
    profile: DatasetProfile,
    hypothesis_id: int,
    sandbox: SandboxSession,
    max_retries: int = config.MAX_RETRIES_PER_STEP,
) -> tuple[list[StepAttempt], list[CallMetrics]]:
    cache = get_cache()
    attempts: list[StepAttempt] = []
    metrics: list[CallMetrics] = []
    failure_history: list[str] = []
    fingerprint = profile.content_fingerprint

    for attempt_num in range(1, max_retries + 1):
        cache_match_type = "none"
        generation_degraded = False
        code: str

        if attempt_num == 1:
            exact = cache.get(fingerprint, step.description, step.backend)
            if exact is not None:
                code = exact["code"]
                cache_match_type = "exact"
                exec_result = exact["result"]
                attempt = StepAttempt(
                    step_id=step.id, hypothesis_id=hypothesis_id, attempt_number=attempt_num,
                    code=code, cache_match_type="exact", **_attempt_fields(exec_result, False),
                )
                attempts.append(attempt)
                return attempts, metrics  # exact hit reuses the whole verified result directly

            semantic = cache.semantic_lookup(fingerprint, step.description, step.backend)
            if semantic is not None:
                code = semantic["code"]
                cache_match_type = "semantic"
            else:
                code, m = _generate_code(step, profile, failure_history)
                metrics.append(m)
                generation_degraded = m.degraded_mode
        else:
            code, m = _generate_code(step, profile, failure_history)
            metrics.append(m)
            generation_degraded = m.degraded_mode

        exec_result = sandbox.execute(code, step.id)
        plan_issue = _detect_plan_issue(exec_result)
        attempt = StepAttempt(
            step_id=step.id, hypothesis_id=hypothesis_id, attempt_number=attempt_num,
            code=code, cache_match_type=cache_match_type, plan_issue_detected=plan_issue,
            **_attempt_fields(exec_result, generation_degraded),
        )
        attempts.append(attempt)

        if exec_result.get("policy_denied"):
            failure_history.append(exec_result.get("policy_denial_reason", "policy denied"))
            continue

        if exec_result.get("error"):
            failure_history.append(exec_result["error"][:500])
            continue

        if _is_soft_failure(exec_result):
            failure_history.append("result was empty or missing -- recompute with a clearer approach")
            continue

        # Clean success: cache it (cache_match_type "semantic" still gets
        # re-verified fresh here before being cached as this step's own
        # exact entry) and return -- but NEVER cache a result produced by a
        # degraded-mode code-generation call (heuristic stub / local
        # fallback). Caching it would let a placeholder answer get served
        # back as an "exact match, reuse directly" hit indefinitely, even
        # after a real provider is configured -- exactly the bug this
        # guards against.
        if not generation_degraded:
            cache.put(fingerprint, step.description, step.backend, code, exec_result)
        return attempts, metrics

    return attempts, metrics


def _attempt_fields(exec_result: dict, generation_degraded: bool) -> dict:
    return {
        "stdout": exec_result.get("stdout", ""),
        "error": exec_result.get("error"),
        "result_repr": exec_result.get("result_repr"),
        "chart_path": exec_result.get("chart_path"),
        "rows_in": exec_result.get("rows_in", 0),
        "rows_out": exec_result.get("rows_out", 0),
        "columns_touched": exec_result.get("columns_touched", []),
        "static_check_rejected": bool(exec_result.get("policy_denied") and exec_result.get("policy_denial_reason", "").startswith("disallowed")),
        "policy_denied": exec_result.get("policy_denied", False),
        "policy_denial_reason": exec_result.get("policy_denial_reason"),
        "degraded_mode": generation_degraded or exec_result.get("degraded_mode", False),
        "sandbox_runtime": exec_result.get("sandbox_runtime", "primary"),
    }


def _detect_plan_issue(exec_result: dict) -> str | None:
    stdout = exec_result.get("stdout") or ""
    for line in stdout.splitlines():
        if line.strip().startswith("PLAN_ISSUE:"):
            return line.strip()[len("PLAN_ISSUE:"):].strip()
    return None


def _is_soft_failure(exec_result: dict) -> bool:
    if exec_result.get("error"):
        return False
    result_repr = exec_result.get("result_repr")
    return result_repr is None or result_repr in ("None", "''", '""', "[]", "{}")

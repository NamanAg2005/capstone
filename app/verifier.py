"""
Verification (Section 9): a verified answer has passed three distinct
checks, plus a conditional fourth.

1. A dedicated call confirms the synthesized answer actually addresses the
   question asked ("the code ran" != "this answers the question").
2. If the question was ambiguous enough to produce >1 plan, the same pass
   compares the resulting answers and either picks the better-supported
   reading or presents both with the ambiguity stated.
3. A scope-of-claim check flags when the question asked for something the
   dataset alone can't establish (external comparison, causal claim,
   out-of-range prediction).
4. If the answer's lineage touches columns carrying data-quality flags, a
   local bootstrap resampling pass computes an actual confidence interval
   instead of a vague caveat; bootstrap only applies to pandas-backend
   steps in this implementation (duckdb views aren't re-bindable to a
   resampled frame without a more involved registration step).
"""
from __future__ import annotations

import re

import numpy as np

from app.llm import call_structured, call_text
from app.models import AnalysisPlan, ConfidenceInterval, DatasetProfile, PlanStep, StepAttempt
from app.sandbox import SandboxSession

_SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "addresses_question": {"type": "boolean"},
        "lineage": {"type": "string"},
    },
    "required": ["answer", "addresses_question", "lineage"],
}

_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "in_scope": {"type": "boolean"},
        "scope_note": {"type": "string"},
    },
    "required": ["in_scope"],
}


def synthesize_answer(question: str, hypothesis_steps: dict[int, list[StepAttempt]]):
    """Build a natural-language answer from the final successful attempt of
    each hypothesis's step chain, and confirm it addresses the question."""
    sections = []
    for hyp_id, attempts in hypothesis_steps.items():
        final_results = [a for a in attempts if a.error is None and not a.policy_denied]
        results_text = "\n".join(f"  step {a.step_id}: {a.result_repr}" for a in final_results[-5:])
        sections.append(f"Hypothesis {hyp_id} results:\n{results_text}")

    system = (
        "You are the Verifier. Given the question and the raw computed results from one or more "
        "candidate analysis plans, write a clear, direct natural-language answer. If there is more "
        "than one hypothesis, either identify the clearly better-supported reading or present both "
        "with the specific ambiguity stated in plain terms. Also state the lineage: which "
        "steps/columns the answer is actually based on. Set addresses_question to false if the "
        "computed results do not actually let you answer what was asked."
    )
    user = f"Question: {question}\n\n" + "\n\n".join(sections)

    fallback = {
        "answer": "Unable to synthesize a verified answer without a model tier available.",
        "addresses_question": False,
        "lineage": "none",
    }
    resp = call_structured(
        tier="verifier", system_prompt=system, user_prompt=user, tool_name="synthesize",
        tool_schema=_SYNTH_SCHEMA, heuristic_fallback=fallback, max_tokens=1200,
    )
    return resp.data, resp.metrics


def scope_check(question: str, profile: DatasetProfile):
    system = (
        "You are the Verifier's scope-of-claim check. Decide whether the question asks for "
        "something the dataset ALONE can actually establish, or whether it reaches beyond that: "
        "a comparison to an external figure not present in the data, a causal claim the data can't "
        "support (correlation only), or a prediction beyond the observed range. If it reaches "
        "beyond the data, set in_scope=false and write a one-sentence scope_note explaining what "
        "part of the question the dataset can't actually support."
    )
    user = f"Dataset tables: {list(profile.tables.keys())}\nQuestion: {question}"
    fallback = {"in_scope": True, "scope_note": ""}
    resp = call_structured(
        tier="verifier", system_prompt=system, user_prompt=user, tool_name="scope_check",
        tool_schema=_SCOPE_SCHEMA, heuristic_fallback=fallback, max_tokens=300,
    )
    note = resp.data.get("scope_note") if not resp.data.get("in_scope", True) else None
    return note, resp.metrics


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(result_repr: str | None) -> float | None:
    if not result_repr:
        return None
    m = _NUMBER_RE.search(result_repr)
    return float(m.group()) if m else None


def compute_confidence_interval(
    sandbox: SandboxSession, final_attempt: StepAttempt, step: PlanStep, n_resamples: int = 30, seed: int = 0,
) -> ConfidenceInterval | None:
    if step.backend != "pandas":
        return None
    point = _extract_number(final_attempt.result_repr)
    if point is None:
        return None

    bootstrap_prefix = (
        "import numpy as _np\n"
        "_rng = _np.random.default_rng({seed})\n"
        "for _k in list(dfs.keys()):\n"
        "    dfs[_k] = dfs[_k].sample(frac=1.0, replace=True, random_state=_rng.integers(0, 1_000_000))\n"
        "    globals()[_k] = dfs[_k]\n"
    )
    samples: list[float] = []
    for i in range(n_resamples):
        wrapped = bootstrap_prefix.format(seed=seed + i) + final_attempt.code
        res = sandbox.execute(wrapped, step.id, timeout_s=10)
        if res.get("error") or res.get("policy_denied"):
            continue
        val = _extract_number(res.get("result_repr"))
        if val is not None:
            samples.append(val)

    if len(samples) < max(5, n_resamples // 3):
        return None  # not enough usable resamples to trust a range

    low, high = np.percentile(samples, [5, 95])
    return ConfidenceInterval(
        point_estimate=point, low=float(low), high=float(high),
        basis=f"bootstrap resampling, n={len(samples)} of {n_resamples}, over rows touching flagged columns",
    )


def lineage_touches_flagged_columns(profile: DatasetProfile, columns_touched: list[str]) -> bool:
    for cols in profile.tables.values():
        for c in cols:
            if c.name in columns_touched and c.quality_flags:
                return True
    return False

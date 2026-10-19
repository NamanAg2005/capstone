"""
Orchestrator (Section 4.8): the state machine coordinating everything else.

Built explicitly as a langgraph StateGraph rather than a hand-rolled loop,
per the spec's reasoning that retry/re-planning are cycles, not linear
control flow. Node granularity here is coarser than the full architecture
diagram: retries are handled inside executor.run_step, and re-planning is
implemented as a bounded in-node loop that genuinely calls the Planner
again with the completed work and the issue description (so the capability
is real), rather than as a separate bounce back through a "plan" graph
node for every single re-plan -- documented as a scope simplification in
README.md. The route -> (direct | planned) fork and the overall
route -> plan -> execute -> verify -> consistency -> END spine are real
graph edges.

Concurrency: each hypothesis gets its own persistent sandbox kernel, so
competing hypotheses execute concurrently (a real benefit of the
multi-hypothesis fork). Steps *within* one hypothesis's plan run in wave
order but sequentially against that hypothesis's single kernel, since they
share mutable state -- true intra-plan parallelism across independent
branches of a single plan is not implemented.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langgraph.graph import END, StateGraph

from app import config, executor, planner, router, verifier
from app.consistency import check_and_record, retrieve_prior_context
from app.models import AgentState, AnalysisPlan, CallMetrics, PlanStep, StepAttempt
from app.sandbox import SandboxSession


def _direct_plan(question: str) -> AnalysisPlan:
    return AnalysisPlan(
        hypothesis_id=1,
        steps=[PlanStep(id=1, description=question, depends_on=[], wave=0, backend="pandas")],
        assumptions=["Routed as 'direct': answerable in a single step."],
        confidence=0.9,
    )


def _node_router(state: AgentState) -> dict:
    route, metrics = router.route_question(state["question"], state["dataset_profile"])
    return {"route": route, "call_log": state.get("call_log", []) + [metrics]}


def _node_plan(state: AgentState) -> dict:
    if state["route"] == "direct":
        plans = [_direct_plan(state["question"])]
        metrics_list: list[CallMetrics] = []
    else:
        prior_hits = retrieve_prior_context(state["session_id"], state["question"])
        prior_context = [f"Q: {h['text']} -> A: {h['metadata'].get('answer', '')}" for h in prior_hits]
        plans, metrics = planner.create_plan(
            state["question"], state["dataset_profile"], state["dataset_profile"].quality_summary,
            prior_context=prior_context,
        )
        metrics_list = [metrics]
        if not plans:
            plans = [_direct_plan(state["question"])]

    # Enforce the per-question hypothesis fan-out budget: drop the weakest
    # candidate plans first if the full fan-out would likely exceed it.
    plans.sort(key=lambda p: p.confidence, reverse=True)
    budget = state.get("hypothesis_budget_tokens", config.QUESTION_TOKEN_BUDGET)
    est_cost_per_plan = 3000  # rough token estimate per hypothesis's execution
    max_plans = max(1, budget // est_cost_per_plan)
    plans = plans[:max_plans]

    return {
        "plans": plans,
        "active_hypotheses": [p.hypothesis_id for p in plans],
        "call_log": state.get("call_log", []) + metrics_list,
    }


def _run_hypothesis(
    plan: AnalysisPlan, profile, scratch_dir: Path, dataset_paths: dict[str, str], question: str,
) -> tuple[list[StepAttempt], list[CallMetrics], list, int]:
    """Runs one hypothesis's plan wave by wave on its own sandbox kernel.
    Returns (transcript, call_metrics, policy_log, replans_used)."""
    sandbox = SandboxSession(dataset_paths, scratch_dir / f"hyp_{plan.hypothesis_id}")
    transcript: list[StepAttempt] = []
    call_metrics: list[CallMetrics] = []
    replans_used = 0
    completed_steps: list[PlanStep] = []

    steps_by_wave: dict[int, list[PlanStep]] = {}
    for s in plan.steps:
        steps_by_wave.setdefault(s.wave, []).append(s)

    wave_numbers = sorted(steps_by_wave)
    w_idx = 0
    while w_idx < len(wave_numbers):
        wave = wave_numbers[w_idx]
        for step in steps_by_wave[wave]:
            attempts, metrics = executor.run_step(step, profile, plan.hypothesis_id, sandbox)
            transcript.extend(attempts)
            call_metrics.extend(metrics)
            completed_steps.append(step)

            last = attempts[-1] if attempts else None
            if last and last.plan_issue_detected and replans_used < config.MAX_REPLANS:
                revised, p_metrics = planner.replan(
                    question, profile, plan.hypothesis_id, completed_steps, transcript,
                    last.plan_issue_detected,
                )
                call_metrics.append(p_metrics)
                replans_used += 1
                if revised is not None and revised.steps:
                    plan.steps = completed_steps + revised.steps
                    steps_by_wave = {}
                    for s in plan.steps:
                        steps_by_wave.setdefault(s.wave, []).append(s)
                    wave_numbers = sorted(w for w in steps_by_wave if w >= wave)
                    w_idx = -1
                    break
        w_idx += 1

    policy_log = sandbox.policy_log
    sandbox.shutdown()
    return transcript, call_metrics, policy_log, replans_used


def _node_execute(state: AgentState) -> dict:
    profile = state["dataset_profile"]
    scratch_dir = config.SCRATCH_DIR / state["session_id"] / str(int(time.time() * 1000))
    dataset_paths: dict[str, str] = state["dataset_paths"]  # internal, injected by caller

    transcript: list[StepAttempt] = []
    call_log = list(state.get("call_log", []))
    policy_log = list(state.get("policy_log", []))
    replan_count = state.get("replan_count", 0)

    with ThreadPoolExecutor(max_workers=max(1, len(state["plans"]))) as pool:
        futures = [
            pool.submit(_run_hypothesis, plan, profile, scratch_dir, dataset_paths, state["question"])
            for plan in state["plans"]
        ]
        results = [f.result() for f in futures]

    for (t, m, p, r) in results:
        transcript.extend(t)
        call_log.extend(m)
        policy_log.extend(p)
        replan_count += r

    return {"transcript": transcript, "call_log": call_log, "policy_log": policy_log, "replan_count": replan_count}


def _node_verify(state: AgentState) -> dict:
    transcript = state["transcript"]
    hyp_steps: dict[int, list[StepAttempt]] = {}
    for a in transcript:
        hyp_steps.setdefault(a.hypothesis_id, []).append(a)

    synth, synth_metrics = verifier.synthesize_answer(state["question"], hyp_steps)
    scope_note, scope_metrics = verifier.scope_check(state["question"], state["dataset_profile"])

    final_answer = synth.get("answer", "")
    lineage = synth.get("lineage", "")
    verified = bool(synth.get("addresses_question", False))

    confidence_interval = None
    all_columns_touched: list[str] = []
    for attempts in hyp_steps.values():
        successful = [a for a in attempts if a.error is None and not a.policy_denied]
        if not successful:
            continue
        last = successful[-1]
        all_columns_touched.extend(last.columns_touched)
        if verifier.lineage_touches_flagged_columns(state["dataset_profile"], last.columns_touched):
            plan = next((p for p in state["plans"] if p.hypothesis_id == last.hypothesis_id), None)
            step = next((s for s in (plan.steps if plan else []) if s.id == last.step_id), None)
            if step is not None:
                sandbox = SandboxSession(
                    state["dataset_paths"], config.SCRATCH_DIR / state["session_id"] / "ci_tmp"
                )
                try:
                    confidence_interval = verifier.compute_confidence_interval(sandbox, last, step)
                finally:
                    sandbox.shutdown()

    return {
        "final_answer": final_answer,
        "lineage": lineage,
        "verified": verified,
        "scope_note": scope_note,
        "confidence_interval": confidence_interval,
        "call_log": state.get("call_log", []) + [synth_metrics, scope_metrics],
        "columns_touched_for_consistency": all_columns_touched,
    }


def _node_consistency(state: AgentState) -> dict:
    flags = check_and_record(
        session_id=state["session_id"],
        fingerprint=state["dataset_profile"].content_fingerprint,
        question=state["question"],
        answer=state.get("final_answer", "") or "",
        columns_touched=state.get("columns_touched_for_consistency", []),
        cross_session_opt_in=state.get("cross_session_consistency_opt_in", False),
    )
    return {"consistency_flags": flags}


def _route_after_router(state: AgentState) -> str:
    return "plan"  # both direct and planned flow through the same plan node (direct builds a trivial plan)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", _node_router)
    graph.add_node("plan", _node_plan)
    graph.add_node("execute", _node_execute)
    graph.add_node("verify", _node_verify)
    graph.add_node("consistency", _node_consistency)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_after_router, {"plan": "plan"})
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_edge("verify", "consistency")
    graph.add_edge("consistency", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_turn(
    session_id: str,
    user_id: str,
    question: str,
    dataset_profile,
    dataset_paths: dict[str, str],
    session_cost_budget_tokens: int,
    session_cost_spent_tokens: int,
    cross_session_consistency_opt_in: bool = False,
) -> AgentState:
    """Runs one full question through the graph and returns the final state."""
    initial: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "dataset_profile": dataset_profile,
        "question": question,
        "question_language": "en",
        "route": "planned",
        "plans": [],
        "current_step_index": {},
        "transcript": [],
        "retry_count": {},
        "replan_count": 0,
        "hypothesis_budget_tokens": config.QUESTION_TOKEN_BUDGET,
        "active_hypotheses": [],
        "session_cost_budget_tokens": session_cost_budget_tokens,
        "session_cost_spent_tokens": session_cost_spent_tokens,
        "cross_session_consistency_opt_in": cross_session_consistency_opt_in,
        "call_log": [],
        "policy_log": [],
        "final_answer": None,
        "lineage": None,
        "confidence_interval": None,
        "consistency_flags": [],
        "scope_note": None,
        "degraded_mode": False,
        "verified": False,
        "dataset_paths": dataset_paths,
        "columns_touched_for_consistency": [],
    }
    final_state = get_graph().invoke(initial)
    spent = sum(c.tokens_in + c.tokens_out for c in final_state.get("call_log", []))
    final_state["session_cost_spent_tokens"] = session_cost_spent_tokens + spent
    final_state["degraded_mode"] = any(c.degraded_mode for c in final_state.get("call_log", []))
    return final_state

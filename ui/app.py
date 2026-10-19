"""
Web interface (Section 4.9). Run with:

    streamlit run ui/app.py

One core interaction -- upload, ask, watch the plan and each step's code/
output/retries stream in, then see the verified answer -- surrounded by
the supporting views the rest of the architecture produces.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, feedback, ingestion, orchestrator, sessions  # noqa: E402
from app.costs import summarize  # noqa: E402
from app.export import save_notebook  # noqa: E402
from app.ingestion import UnsupportedFormatError  # noqa: E402

st.set_page_config(page_title="Autonomous Data-Analysis Agent", layout="wide")

if "session" not in st.session_state:
    st.session_state.session = sessions.get_session_store().create(user_id="local-user")
if "turns" not in st.session_state:
    st.session_state.turns = []
if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = {}

sess = st.session_state.session

st.title("Autonomous Data-Analysis Agent")
if config.DRY_RUN:
    st.warning(
        "No model provider is configured in .env (set the free GROQ_API_KEY). Running in DRY-RUN / heuristic-fallback mode: every model tier uses a "
        "deterministic stub instead of a real LLM call, so answers will be weak. The full "
        "pipeline (ingestion, caching, sandboxed execution, verification structure) still runs "
        "end-to-end."
    )
else:
    st.caption(f"Model provider: {config.PROVIDER}")

tab_analyze, tab_ops = st.tabs(["Analyze", "Operations"])

with tab_analyze:
    st.subheader("1. Upload dataset(s)")
    uploaded_files = st.file_uploader(
        "CSV, Excel, Parquet, or JSON -- one or more tables",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "parquet", "json"],
    )

    if uploaded_files:
        dfs = {}
        rejected = []
        for uf in uploaded_files:
            dest = config.SCRATCH_DIR / sess.session_id / uf.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(uf.getvalue())
            try:
                df = ingestion.load_table(dest)
                name = Path(uf.name).stem
                dfs[name] = df
                st.session_state.uploaded_paths[name] = str(dest)
            except UnsupportedFormatError as e:
                rejected.append((uf.name, str(e)))

        for fname, reason in rejected:
            st.error(f"Rejected '{fname}': {reason}")

        if dfs:
            profile = ingestion.build_dataset_profile(dfs)
            sess.dataset_profile = profile
            sess.dataset_paths = st.session_state.uploaded_paths

            with st.expander("Pre-flight: data-quality & PII summary", expanded=True):
                st.text(profile.quality_summary)
                for table, cols in profile.tables.items():
                    st.markdown(f"**{table}** ({profile.row_counts[table]} rows)")
                    st.dataframe(
                        [
                            {
                                "column": c.name, "dtype": c.dtype, "null_%": c.null_pct,
                                "cardinality": c.cardinality, "pii": c.pii_detected,
                                "quality_flags": ", ".join(c.quality_flags),
                                "sample (masked)": ", ".join(c.sample_values),
                            }
                            for c in cols
                        ],
                        use_container_width=True,
                    )
                if profile.join_candidates:
                    st.markdown("**Join candidates**")
                    for j in profile.join_candidates:
                        st.write(
                            f"{j.left_table}.{j.left_column} <-> {j.right_table}.{j.right_column} "
                            f"(confidence={j.confidence}, validated={j.sample_validated}, auto_confirmed={j.auto_confirmed})"
                        )

            sess.cross_session_opt_in = st.checkbox(
                "Opt in to anonymized cross-session consistency checking for this dataset",
                value=sess.cross_session_opt_in,
                help="Stores only this dataset's fingerprint, which columns a verified answer touched, "
                     "the value, and a timestamp -- never the question text or your identity.",
            )

    st.subheader("2. Ask a question")
    question = st.text_input("Question (any supported language)", key="question_input")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question and sess.dataset_profile is not None:
        allowed, wait_s = sessions.get_session_store().check_rate_limit(sess.user_id)
        if not allowed:
            st.error(f"Rate limit exceeded. Try again in {wait_s:.0f}s.")
        elif sess.cost_spent_tokens > sess.cost_budget_tokens and not st.session_state.get("confirmed_over_budget"):
            st.warning(f"Session cost budget ({sess.cost_budget_tokens} tokens) exceeded.")
            if st.button("Confirm and continue anyway"):
                st.session_state["confirmed_over_budget"] = True
        else:
            with st.spinner("Routing, planning, executing, verifying..."):
                state = orchestrator.run_turn(
                    session_id=sess.session_id, user_id=sess.user_id, question=question,
                    dataset_profile=sess.dataset_profile, dataset_paths=sess.dataset_paths,
                    session_cost_budget_tokens=sess.cost_budget_tokens,
                    session_cost_spent_tokens=sess.cost_spent_tokens,
                    cross_session_consistency_opt_in=sess.cross_session_opt_in,
                )
            sess.cost_spent_tokens = state["session_cost_spent_tokens"]
            st.session_state.turns.append(state)
            sess.turns = st.session_state.turns

    st.subheader("3. Trace, verified answer, and history")
    for i, state in enumerate(reversed(st.session_state.turns)):
        idx = len(st.session_state.turns) - i
        with st.container(border=True):
            st.markdown(f"**Q{idx}: {state['question']}**")
            if state.get("degraded_mode"):
                st.info("This run used a fallback (degraded) model tier for at least one call. "
                        "Treat the answer as weaker than normal.")

            plans = state.get("plans", [])
            cols = st.columns(len(plans)) if len(plans) > 1 else [st.container()]
            for plan, col in zip(plans, cols):
                with col:
                    st.caption(f"Hypothesis {plan.hypothesis_id} (confidence {plan.confidence})")
                    if plan.assumptions:
                        st.caption("Assumptions: " + "; ".join(plan.assumptions))
                    hyp_attempts = [a for a in state.get("transcript", []) if a.hypothesis_id == plan.hypothesis_id]
                    for a in hyp_attempts:
                        label = f"step {a.step_id}, attempt {a.attempt_number}"
                        if a.error:
                            label += " -- error, retried" if a.attempt_number < config.MAX_RETRIES_PER_STEP else " -- failed"
                        elif a.plan_issue_detected:
                            label += " -- plan issue detected, re-planned"
                        elif a.policy_denied:
                            label += " -- policy denied"
                        else:
                            label += f" -- ok ({a.cache_match_type} cache)" if a.cache_match_type != "none" else " -- ok"
                        with st.expander(label):
                            st.code(a.code, language="python")
                            if a.stdout:
                                st.text(a.stdout)
                            if a.error:
                                st.error(a.error)
                            if a.result_repr:
                                st.write(f"Result: {a.result_repr}")
                            if a.sandbox_runtime == "fallback":
                                st.caption("Ran on the fallback (less isolated) sandbox runtime.")

            st.markdown(f"**Verified answer:** {state.get('final_answer', '(none)')}")
            st.caption(f"Lineage: {state.get('lineage', '')}")
            ci = state.get("confidence_interval")
            if ci:
                st.caption(f"Confidence interval: {ci.low:.3g} - {ci.high:.3g} ({ci.basis})")
            if state.get("scope_note"):
                st.warning(f"Scope note: {state['scope_note']}")
            for flag in state.get("consistency_flags", []):
                st.warning(f"Consistency flag ({flag.scope}): {flag.description}")

            cost = summarize(state.get("call_log", []))
            st.caption(
                f"Cost: {cost['total_tokens_in']} in / {cost['total_tokens_out']} out tokens, "
                f"{cost['total_latency_ms']}ms total latency, "
                f"{state.get('replan_count', 0)} re-plan(s)."
            )

            answer_id = f"{sess.session_id}:{idx}"
            fb_col1, fb_col2 = st.columns(2)
            if fb_col1.button("👍", key=f"up_{idx}"):
                feedback.submit_feedback(
                    answer_id, "up", state["question"], state.get("final_answer", ""),
                    state.get("transcript", []), sess.session_id, sess.user_id,
                    sess.dataset_profile, sess.dataset_paths,
                )
                st.success("Marked correct -- promoted as a benchmark candidate.")
            if fb_col2.button("👎", key=f"down_{idx}"):
                with st.spinner("Running independent re-derivation..."):
                    fb = feedback.submit_feedback(
                        answer_id, "down", state["question"], state.get("final_answer", ""),
                        state.get("transcript", []), sess.session_id, sess.user_id,
                        sess.dataset_profile, sess.dataset_paths,
                    )
                st.info("Logged to the review queue with an independent re-derivation attached.")

    if st.session_state.turns:
        if st.button("Export session as notebook"):
            out_path = config.SCRATCH_DIR / sess.session_id / f"session_{int(time.time())}.ipynb"
            save_notebook(st.session_state.turns, str(out_path))
            st.success(f"Saved to {out_path}")
            st.download_button("Download notebook", data=out_path.read_bytes(), file_name=out_path.name)

with tab_ops:
    from app import ops

    st.subheader("Operations view")
    if st.button("Refresh"):
        st.rerun()
    report = ops.full_report()
    st.json(report)

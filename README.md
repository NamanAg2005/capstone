# Autonomous Data-Analysis Agent

A working implementation of the architecture in `docs/architecture-spec.pdf`:
upload one or more tabular datasets, ask a question in plain English, and
get back a **verified** answer — the reasoning, the code that produced it,
the specific rows/columns behind it, a confidence range when the data
quality warrants one, and an explicit note whenever the question asks for
something the data alone can't support.

It is a real agent loop (Router → Planner → Executor → Verify →
Consistency check), not a single-shot code generator: it retries failed
code, re-plans when execution reveals a wrong assumption, caches repeated
work, remembers earlier questions in the same session, checks new answers
against prior ones, and falls back to a degraded mode (clearly marked)
rather than simply failing when a model call isn't available.

## Quick start

```bash
# 1. Create and activate the virtual environment (already done if you ran
#    this from the setup script — recreate with:)
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash / WSL
# venv\Scripts\activate.bat         # Windows cmd.exe
# venv\Scripts\Activate.ps1         # PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
#   get a free key at https://console.groq.com/keys -- no card required.

# 4. Run the web UI
streamlit run ui/app.py
```

Without a Groq key, the app still runs end-to-end in **DRY-RUN mode**: every
model tier (Router/Planner/Executor/Verifier) uses a deterministic
heuristic stub instead of a real LLM call, so you can exercise ingestion,
caching, the sandbox, the orchestration graph, and the UI with zero cost
and zero network access — answers will just be weak/placeholder text,
and every such run is marked `degraded_mode=True` so it's never mistaken
for a real result.

**Why Groq:** Groq hosts fast open models on a genuinely free tier with
rate limits, so the whole project runs at zero cost. It uses real
structured tool-calling through an OpenAI-compatible client, the same
client the project uses for its local-inference fallback. Expect an
occasional retry on complex multi-step plans or when the free-tier rate
limit is hit; the whole pipeline (caching, sandboxing, verification,
consistency checks) applies either way.

## What's implemented vs. simplified

This is a genuine, runnable implementation of the architecture's core
loop and every section in the spec has a real counterpart in the code —
but a few pieces are deliberately scoped down from "enterprise deployment"
to "runs correctly on one Windows machine with no extra infrastructure."
Each simplification is also called out in a comment at the top of the
relevant module.

| Spec section | Implemented as | Simplification |
|---|---|---|
| Ingestion & governance (`app/ingestion.py`, `app/pii.py`) | Real: modality gate, per-column profiling, regex-based PII detection + masking, data-quality scan, name/overlap join inference with sample validation, content-fingerprint hashing | PII detection is regex/heuristic (emails, phones, SSN-shaped, name-shaped columns), not a full NLP PII model |
| Vector store (`app/vectorstore.py`) | Real: SQLite-backed store with 4 collections (schema index, session memory, semantic cache, cross-session index), add/query/filter | Embeddings are a local hashing bag-of-words vectorizer, not a trained embedding model — chosen deliberately so similarity search needs **zero network calls**, keeping "only egress is model API calls" true even for local search. Fine at single-session/small-deployment scale; not a production ANN index. |
| Router / Planner / Executor / Verifier (`app/router.py`, `planner.py`, `executor.py`, `verifier.py`) | Real: Groq tool-use (function calling) for structured plans/code, multi-hypothesis fan-out, wave-based dependency ordering, re-planning on detected plan issues | — |
| Provider resilience (`app/llm.py`) | Real: primary provider is Groq's free tier, using real structured tool-calling; circuit breaker per tier; falls back to an OpenAI-compatible local endpoint if `LOCAL_MODEL_BASE_URL` is set, else a heuristic stub | No bundled local model for the last-resort fallback — you supply your own (Ollama/vLLM/LM Studio) if you want a real model there rather than the stub |
| Sandboxed execution (`app/sandbox.py`, `sandbox_kernel.py`, `static_check.py`) | Real: separate OS process per session, persistent kernel (pandas + DuckDB), AST static check, network-call monkeypatch, wall-clock timeout + memory ceiling enforced by the parent process via `psutil`, automatic fallback to an in-process restricted-exec runtime if the subprocess sandbox can't start | No container/VM-level isolation (no Docker requirement) — isolation is OS-process + restricted-builtins + resource polling, appropriate for a single-user dev box, not a multi-tenant cloud deployment. `resource.setrlimit` isn't available on Windows, hence the `psutil`-polling approach. |
| Caching (`app/cache.py`) | Real two-tier cache: SQLite exact-match keyed on (fingerprint, step description, backend); semantic hits reuse code only and are always re-executed/re-verified | — |
| Orchestrator (`app/orchestrator.py`) | Real `langgraph` `StateGraph` with the router→plan→execute→verify→consistency spine and real cycles (retry inside the executor, re-planning as a genuine second Planner call) | Node granularity is coarser than every arrow in the architecture diagram — re-planning is a bounded loop inside the "execute" node rather than a separate graph bounce back to a "plan" node for each individual re-plan. The *capability* (Planner called again with completed work + issue description, up to `MAX_REPLANS`) is real. |
| Verification (`app/verifier.py`) | Real: LLM call confirms the answer addresses the question, multi-hypothesis comparison, scope-of-claim check, and a genuine bootstrap confidence interval (resamples rows and re-executes the step's actual code N times) when lineage touches quality-flagged columns | Bootstrap CI only covers pandas-backend steps (DuckDB views aren't re-bound to a resampled frame in this implementation) |
| Session memory & consistency (`app/consistency.py`) | Real: session-scoped vector memory for follow-up continuity, always-on session consistency check, opt-in anonymized cross-session index (fingerprint + column signature + value + timestamp only, never text or identity) | — |
| Multi-user & rate limiting (`app/sessions.py`) | Real in-process session isolation, token-bucket-style per-user rate limiter, per-session cost budget | Not a production auth provider — tokens are opaque bearer strings minted per "login," not persisted across restarts, no password/OAuth flow |
| Evaluation harness (`eval/`) | Real: runs the fixed benchmark set through the actual `orchestrator.run_turn`, reports pass rate, per-category coverage gaps, retry/re-plan stats, excludes degraded runs | Benchmark set is small (8 seed cases) — meant to be grown, not exhaustive |
| Feedback / review queue (`app/feedback.py`) | Real: thumbs-down triggers an independent fresh re-derivation (new session, no access to the flagged transcript) and auto-drafts a candidate fix on disagreement; thumbs-up promotes a benchmark candidate | Queue is a local JSONL file, not a database with a review UI |
| Operations view (`app/ops.py`) | Real: aggregates model-tier fallback status, sandbox health, review-queue backlog, coverage gaps, and session cost stats into one view (the "Operations" tab in the UI) | — |
| Deployment / elastic sandbox pool (spec §4.10) | Not implemented | This would need container orchestration (e.g. Kubernetes) to do properly; out of scope for a single-machine project. The sandbox itself is still a real isolated OS process per session. |
| Translation (`app/translation.py`) | Real: `langdetect` for language ID, translation routed through the same Groq model client (so translation doesn't add a second network egress point) | — |

## Project layout

```
app/              Core library: every component in Section 4 of the spec
  config.py         Environment/config loading
  models.py         Pydantic data contracts (Section 7) + AgentState
  ingestion.py       Modality gate, profiling, PII scan, quality scan, joins
  pii.py             PII pattern detection + masking
  vectorstore.py     Local embedded vector store (4 collections)
  cache.py           Two-tier exact/semantic cache
  llm.py             Model client, circuit breaker, tiered fallback
  router.py          Direct vs. planned classification
  planner.py         Multi-hypothesis structured planning + re-planning
  executor.py        Cache check → codegen → sandbox → retry loop
  static_check.py    AST-based pre-execution code policy
  sandbox.py         Parent-side sandbox process management + resource limits
  sandbox_kernel.py  Child-process persistent kernel (pandas/DuckDB)
  verifier.py        Answer verification, scope check, bootstrap CI
  consistency.py     Session memory + session/cross-session consistency
  orchestrator.py    langgraph StateGraph tying it all together
  sessions.py        Multi-user session isolation + rate limiting
  feedback.py        Review queue + independent re-derivation
  export.py          Notebook export
  costs.py           Cost/latency summarization
  ops.py             Operations-view aggregation
ui/app.py         Streamlit web interface
eval/             Evaluation harness + benchmark set
sample_data/      Two linked sample tables (orders, customers) for trying it
tests/            pytest suite (ingestion, static check, sandbox, cache, vectorstore)
docs/architecture-spec.pdf   The original architecture document
```

## Running things

```bash
# Web UI
streamlit run ui/app.py

# Test suite (no API key needed — covers ingestion, static analysis,
# the real sandboxed subprocess, caching, vector store)
python -m pytest tests/ -v

# Evaluation harness against the seed benchmark set
python -m eval.run_eval
```

Try it with the bundled sample data: upload both
`sample_data/orders.csv` and `sample_data/customers.csv`, then ask things
like *"How many orders came from customers in the West region?"* (exercises
join inference) or *"What's the best-selling product?"* (ambiguous enough
to sometimes fork into competing hypotheses).

## Notes on what was verified in this environment

- The full test suite (24 tests) passes, including a real sandboxed
  subprocess: static-check rejection, persistent kernel state across
  calls, a DuckDB query, a captured runtime exception, and an actual
  wall-clock timeout that kills and reports a hung process.
- The evaluation harness was run end-to-end in DRY-RUN mode (no API key)
  against all 8 benchmark cases to confirm the full `langgraph` graph —
  routing, planning, concurrent multi-hypothesis execution, re-planning,
  verification, and consistency checking — completes without error and
  correctly marks every DRY-RUN answer as `degraded_mode` (so it's
  excluded from the pass-rate score, exactly as the spec requires).
- The Streamlit UI was booted headlessly and confirmed to serve HTTP 200.
- **Not yet verified**: actual answer *accuracy* against the benchmark
  set, since that requires the free `GROQ_API_KEY` which wasn't available in this environment. Set
  one in `.env` and re-run `python -m eval.run_eval` to see real pass
  rates.

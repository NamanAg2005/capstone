"""Central configuration, loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_STORE_DIR = ROOT_DIR / "data_store"
SCRATCH_DIR = DATA_STORE_DIR / "scratch"
SESSIONS_DIR = DATA_STORE_DIR / "sessions"
CACHE_DB_PATH = DATA_STORE_DIR / "cache.sqlite3"
VECTOR_DB_PATH = DATA_STORE_DIR / "vectors.sqlite3"
CROSS_SESSION_DB_PATH = DATA_STORE_DIR / "cross_session.sqlite3"
OPS_LOG_DB_PATH = DATA_STORE_DIR / "ops_log.sqlite3"

for d in (DATA_STORE_DIR, SCRATCH_DIR, SESSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Groq: a free-tier, OpenAI-compatible provider hosting fast open models.
# It is the only primary provider -- every tier (Router/Planner/Executor/
# Verifier) calls Groq via the same OpenAI-compatible client used for
# LOCAL_MODEL_BASE_URL below, just pointed at Groq's hosted endpoint.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
GROQ_ROUTER_MODEL = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")
GROQ_PLANNER_MODEL = os.getenv("GROQ_PLANNER_MODEL", "openai/gpt-oss-120b")
GROQ_EXECUTOR_MODEL = os.getenv("GROQ_EXECUTOR_MODEL", "openai/gpt-oss-20b")
GROQ_VERIFIER_MODEL = os.getenv("GROQ_VERIFIER_MODEL", "openai/gpt-oss-120b")

# "groq" when a key is present, otherwise "none" (DRY-RUN heuristic mode).
PROVIDER = "groq" if GROQ_API_KEY else "none"

# Local-inference fallback (Ollama / vLLM / LM Studio, etc): a LAST-resort,
# genuinely degraded path used only if the configured primary provider's
# circuit breaker trips. Distinct from Groq-as-primary above.
LOCAL_MODEL_BASE_URL = os.getenv("LOCAL_MODEL_BASE_URL", "").strip()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "").strip()

SESSION_TOKEN_BUDGET = int(os.getenv("SESSION_TOKEN_BUDGET", "200000"))
QUESTION_TOKEN_BUDGET = int(os.getenv("QUESTION_TOKEN_BUDGET", "20000"))

SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "20"))
SANDBOX_MEMORY_MB = int(os.getenv("SANDBOX_MEMORY_MB", "512"))

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

MAX_RETRIES_PER_STEP = 3
MAX_REPLANS = 2
SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}
SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "hi", "pt", "it", "zh-cn", "ja"}

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
CIRCUIT_BREAKER_WINDOW_SECONDS = 120

DRY_RUN = PROVIDER == "none"

"""
Two-tier caching layer (Section 4.7).

Exact-match tier: local key-value store keyed on
hash(content_fingerprint, step_description, backend), storing the full
successful StepAttempt result. A semantic-match hit (via the vector store's
semantic_cache collection) only ever reuses *code*, never a stored result --
the executor re-runs and re-verifies it fresh, so a fuzzy match can save a
generation call but never produce an unchecked answer.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from app.vectorstore import get_store


def _key(fingerprint: str, description: str, backend: str) -> str:
    raw = f"{fingerprint}:{description.strip().lower()}:{backend}"
    return hashlib.sha256(raw.encode()).hexdigest()


class ExactCache:
    def __init__(self, db_path: Path):
        self._lock = threading.Lock()
        # Shared across hypothesis worker threads; see vectorstore.py note.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exact_cache (
                cache_key TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                description TEXT NOT NULL,
                backend TEXT NOT NULL,
                code TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, fingerprint: str, description: str, backend: str) -> dict | None:
        key = _key(fingerprint, description, backend)
        with self._lock:
            row = self._conn.execute(
                "SELECT code, result_json FROM exact_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        code, result_json = row
        result = json.loads(result_json)
        if result.get("degraded_mode"):
            # Defense in depth: a degraded-mode (heuristic/local-fallback)
            # result should never have been written here (put() now
            # refuses to store one -- see executor.py), but if an older
            # entry from before that guard still exists, treat it as a
            # miss rather than serving a placeholder answer forever.
            return None
        return {"code": code, "result": result}

    def put(self, fingerprint: str, description: str, backend: str, code: str, result: dict) -> None:
        if result.get("degraded_mode"):
            return  # never persist a degraded-mode result as if it were verified
        key = _key(fingerprint, description, backend)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO exact_cache VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, fingerprint, description, backend, code, json.dumps(result), time.time()),
            )
            self._conn.commit()
        # Also index into the semantic cache so near-miss descriptions can
        # find this step's *code* even when the exact key misses.
        get_store().add(
            "semantic_cache",
            key,
            description,
            {"fingerprint": fingerprint, "backend": backend, "code": code},
        )

    def semantic_lookup(self, fingerprint: str, description: str, backend: str, min_similarity: float = 0.82) -> dict | None:
        hits = get_store().query(
            "semantic_cache",
            description,
            top_k=1,
            metadata_filter={"fingerprint": fingerprint, "backend": backend},
            min_similarity=min_similarity,
        )
        if not hits:
            return None
        return {"code": hits[0]["metadata"]["code"], "similarity": hits[0]["similarity"]}


_cache: ExactCache | None = None


def get_cache() -> ExactCache:
    global _cache
    if _cache is None:
        from app import config

        _cache = ExactCache(config.CACHE_DB_PATH)
    return _cache

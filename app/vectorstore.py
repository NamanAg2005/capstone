"""
Embedded local vector store (Section 4.2).

Holds four collections: schema_index, session_memory, semantic_cache, and
cross_session_index. Implemented as a lightweight hashing-vectorizer +
cosine-similarity store backed by SQLite, rather than a third-party vector
DB / embedding model: nothing here needs network access, which keeps the
"only egress is model API calls" principle intact even for local similarity
search, and keeps the project runnable with zero extra downloads.

This is a deliberate scope simplification vs. a production ANN index --
fine at the row counts a single session or small deployment produces, but
not a drop-in substitute for a real vector database at large scale.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path

VECTOR_DIM = 256
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _embed(text: str) -> list[float]:
    vec = [0.0] * VECTOR_DIM
    tokens = _TOKEN_RE.findall(text.lower())
    for tok in tokens:
        idx = hash(tok) % VECTOR_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        # This store is shared across hypothesis worker threads (concurrent
        # plan execution), so the connection must tolerate cross-thread use;
        # the lock below serializes actual access to avoid interleaved calls.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (collection, doc_id)
            )
            """
        )
        self._conn.commit()

    def add(self, collection: str, doc_id: str, text: str, metadata: dict | None = None) -> None:
        vec = _embed(text)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?, ?, ?)",
                (collection, doc_id, text, json.dumps(vec), json.dumps(metadata or {}), time.time()),
            )
            self._conn.commit()

    def query(
        self,
        collection: str,
        text: str,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        q_vec = _embed(text)
        with self._lock:
            rows = self._conn.execute(
                "SELECT doc_id, text, vector, metadata, created_at FROM vectors WHERE collection = ?",
                (collection,),
            ).fetchall()
        scored = []
        for doc_id, doc_text, vec_json, meta_json, created_at in rows:
            meta = json.loads(meta_json)
            if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
            sim = _cosine(q_vec, json.loads(vec_json))
            if sim >= min_similarity:
                scored.append(
                    {"doc_id": doc_id, "text": doc_text, "similarity": sim, "metadata": meta, "created_at": created_at}
                )
        scored.sort(key=lambda r: r["similarity"], reverse=True)
        return scored[:top_k]

    def delete_collection_scope(self, collection: str, metadata_filter: dict) -> None:
        rows = self._conn.execute(
            "SELECT doc_id, metadata FROM vectors WHERE collection = ?", (collection,)
        ).fetchall()
        for doc_id, meta_json in rows:
            meta = json.loads(meta_json)
            if all(meta.get(k) == v for k, v in metadata_filter.items()):
                self._conn.execute(
                    "DELETE FROM vectors WHERE collection = ? AND doc_id = ?", (collection, doc_id)
                )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        from app import config

        _store = VectorStore(config.VECTOR_DB_PATH)
    return _store

"""
Session memory & consistency checking (Sections 4.2, 4.8).

Session memory: every verified question/answer is embedded into the
`session_memory` vector collection so a follow-up question in the same
session can retrieve prior context instead of starting cold.

Consistency: after every verified answer, it's checked against this
session's own prior answers with overlapping lineage (always, no opt-in --
never leaves the session) and, if the user opted in for this dataset,
against the anonymized cross-session index as well. A genuine mismatch
either way is surfaced as a flag, not silently resolved, since a real
difference in filters/assumptions between two questions can legitimately
explain it.

The cross-session index stores only: a dataset's content fingerprint, a
normalized signature of which columns/aggregation a verified answer's
lineage touched, the resulting value, and a timestamp -- never the question
text, never a user or session identifier.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time

from app import config
from app.models import ConsistencyFlag
from app.vectorstore import get_store

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TOLERANCE_PCT = 2.0


def _extract_number(text: str | None) -> float | None:
    if not text:
        return None
    m = _NUMBER_RE.search(text)
    return float(m.group()) if m else None


def _signature(columns_touched: list[str]) -> str:
    return ",".join(sorted(set(columns_touched)))


def remember_session_turn(session_id: str, fingerprint: str, question: str, answer: str, columns_touched: list[str], numeric_value: float | None) -> None:
    get_store().add(
        "session_memory",
        doc_id=f"{session_id}:{time.time()}",
        text=question,
        metadata={
            "session_id": session_id, "fingerprint": fingerprint, "answer": answer,
            "columns_touched": columns_touched, "numeric_value": numeric_value,
        },
    )


def retrieve_prior_context(session_id: str, question: str, top_k: int = 3) -> list[dict]:
    hits = get_store().query("session_memory", question, top_k=top_k, metadata_filter={"session_id": session_id})
    return hits


def check_session_consistency(session_id: str, fingerprint: str, answer: str, columns_touched: list[str]) -> ConsistencyFlag | None:
    numeric_value = _extract_number(answer)
    if numeric_value is None or not columns_touched:
        return None
    hits = get_store().query(
        "session_memory", " ".join(columns_touched), top_k=10,
        metadata_filter={"session_id": session_id, "fingerprint": fingerprint},
    )
    for hit in hits:
        meta = hit["metadata"]
        prior_cols = set(meta.get("columns_touched") or [])
        overlap = prior_cols & set(columns_touched)
        if not overlap:
            continue
        prior_val = meta.get("numeric_value")
        if prior_val is None:
            continue
        if prior_val == 0:
            continue
        pct_diff = abs(numeric_value - prior_val) / abs(prior_val) * 100
        if pct_diff > _TOLERANCE_PCT:
            return ConsistencyFlag(
                scope="session", overlap_columns=sorted(overlap),
                description=(
                    f"This answer ({numeric_value}) differs by {pct_diff:.1f}% from an earlier "
                    f"answer in this session ({prior_val}) that touched the same column(s): "
                    f"{', '.join(sorted(overlap))}. This may be legitimate if filters/assumptions "
                    f"differ between the two questions."
                ),
            )
    return None


class CrossSessionIndex:
    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cross_session (
                fingerprint TEXT NOT NULL,
                signature TEXT NOT NULL,
                value REAL NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(self, fingerprint: str, columns_touched: list[str], value: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO cross_session VALUES (?, ?, ?, ?)",
                (fingerprint, _signature(columns_touched), value, time.time()),
            )
            self._conn.commit()

    def check(self, fingerprint: str, columns_touched: list[str], value: float) -> ConsistencyFlag | None:
        sig = _signature(columns_touched)
        with self._lock:
            rows = self._conn.execute(
                "SELECT value FROM cross_session WHERE fingerprint = ? AND signature = ?",
                (fingerprint, sig),
            ).fetchall()
        for (prior_val,) in rows:
            if prior_val == 0:
                continue
            pct_diff = abs(value - prior_val) / abs(prior_val) * 100
            if pct_diff > _TOLERANCE_PCT:
                return ConsistencyFlag(
                    scope="cross_session", overlap_columns=columns_touched,
                    description=(
                        f"This answer ({value}) differs by {pct_diff:.1f}% from a verified answer "
                        f"produced in a different session against the same dataset, touching the "
                        f"same columns. Fully anonymized comparison -- no question text or session "
                        f"identity involved."
                    ),
                )
        return None


_cross_session_index: CrossSessionIndex | None = None


def get_cross_session_index() -> CrossSessionIndex:
    global _cross_session_index
    if _cross_session_index is None:
        _cross_session_index = CrossSessionIndex(config.CROSS_SESSION_DB_PATH)
    return _cross_session_index


def check_and_record(
    session_id: str, fingerprint: str, question: str, answer: str,
    columns_touched: list[str], cross_session_opt_in: bool,
) -> list[ConsistencyFlag]:
    flags: list[ConsistencyFlag] = []
    numeric_value = _extract_number(answer)

    session_flag = check_session_consistency(session_id, fingerprint, answer, columns_touched)
    if session_flag:
        flags.append(session_flag)

    if cross_session_opt_in and numeric_value is not None and columns_touched:
        idx = get_cross_session_index()
        cross_flag = idx.check(fingerprint, columns_touched, numeric_value)
        if cross_flag:
            flags.append(cross_flag)
        idx.record(fingerprint, columns_touched, numeric_value)

    remember_session_turn(session_id, fingerprint, question, answer, columns_touched, numeric_value)
    return flags

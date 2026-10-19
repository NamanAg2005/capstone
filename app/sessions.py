"""
Multi-user support (Section 4.10): a lightweight auth layer assigning each
user their own isolated session -- own vector-store scope (via
session_id-filtered queries), own cache scope (via dataset fingerprint,
naturally shared across users on purpose so repeated public datasets are
cheap -- see note below), and own running cost total. Rate limiting is a
distinct control from the token-cost budget: it bounds how *often* a user
can call, independent of spend.

Scope note: the exact-match code cache (cache.py) is intentionally scoped
by dataset fingerprint rather than by user, since cached *code* for a given
dataset+step is safe to reuse across users and is always re-verified fresh
on a semantic hit -- this does not leak one user's question text or answer
to another, only previously-generated, re-executed code. Session memory
and the review queue are scoped per session_id / per user, as the spec's
"nothing about one user's data, questions, or spend is visible to another"
requires.

This module implements real in-process session isolation and a real
sliding-window rate limiter; it does not implement a production auth
provider (password hashing, OAuth, etc.) -- tokens here are opaque
bearer strings minted on "login" and are not persisted across restarts.
"""
from __future__ import annotations

import secrets
import time
import uuid
from dataclasses import dataclass, field

from app import config


@dataclass
class Session:
    session_id: str
    user_id: str
    token: str
    created_at: float = field(default_factory=time.time)
    cost_spent_tokens: int = 0
    cost_budget_tokens: int = config.SESSION_TOKEN_BUDGET
    dataset_paths: dict[str, str] = field(default_factory=dict)
    dataset_profile: object = None
    cross_session_opt_in: bool = False
    turns: list = field(default_factory=list)


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, list[float]] = {}

    def allow(self, user_id: str) -> tuple[bool, float]:
        now = time.time()
        window = [t for t in self._hits.get(user_id, []) if now - t < 60]
        if len(window) >= self.per_minute:
            wait_s = 60 - (now - window[0])
            self._hits[user_id] = window
            return False, max(0.0, wait_s)
        window.append(now)
        self._hits[user_id] = window
        return True, 0.0


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self.rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)

    def create(self, user_id: str) -> Session:
        token = secrets.token_urlsafe(24)
        sess = Session(session_id=str(uuid.uuid4()), user_id=user_id, token=token)
        self._sessions[sess.session_id] = sess
        return sess

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def check_rate_limit(self, user_id: str) -> tuple[bool, float]:
        return self.rate_limiter.allow(user_id)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store

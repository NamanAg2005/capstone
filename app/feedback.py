"""
Feedback & review queue (Sections 5.2, 7).

Every thumbs-down is logged with its full transcript, and the harness
automatically follows up with an independent re-derivation -- a fresh
planning-and-execution run with no access to the original flagged
transcript. If the independent answer agrees with the one flagged, the
entry is marked for a developer to investigate why it was flagged rather
than assumed wrong; if it disagrees, the fresh answer becomes an
auto-drafted candidate correct answer attached to the entry, reducing the
developer's task to a single confirm-or-reject decision.

Thumbs-up on a verified answer promotes the question/plan/answer triple
into a candidate benchmark case for a developer to review and promote.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from app import config, orchestrator
from app.models import Feedback, ReviewQueueEntry, StepAttempt

REVIEW_QUEUE_PATH = config.DATA_STORE_DIR / "review_queue.jsonl"
BENCHMARK_CANDIDATES_PATH = config.DATA_STORE_DIR / "benchmark_candidates.jsonl"


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def submit_feedback(
    answer_id: str,
    rating: str,
    question: str,
    final_answer: str,
    transcript: list[StepAttempt],
    session_id: str,
    user_id: str,
    dataset_profile,
    dataset_paths: dict[str, str],
) -> Feedback:
    promoted = False
    added_to_queue = False

    if rating == "up":
        _append_jsonl(
            BENCHMARK_CANDIDATES_PATH,
            {
                "answer_id": answer_id, "question": question, "expected_answer": final_answer,
                "category": "user_promoted", "created_at": time.time(),
            },
        )
        promoted = True

    elif rating == "down":
        independent_state = orchestrator.run_turn(
            session_id=f"review-{uuid.uuid4().hex[:8]}",  # fresh session: no access to original transcript
            user_id=user_id,
            question=question,
            dataset_profile=dataset_profile,
            dataset_paths=dataset_paths,
            session_cost_budget_tokens=config.SESSION_TOKEN_BUDGET,
            session_cost_spent_tokens=0,
        )
        independent_answer = independent_state.get("final_answer") or ""
        agrees = _answers_agree(final_answer, independent_answer)

        entry = ReviewQueueEntry(
            answer_id=answer_id,
            original_transcript=transcript,
            independent_answer=independent_answer,
            agrees_with_original=agrees,
            candidate_correct_answer=None if agrees else independent_answer,
            status="pending_confirmation",
        )
        _append_jsonl(REVIEW_QUEUE_PATH, entry.model_dump())
        added_to_queue = True

    return Feedback(answer_id=answer_id, rating=rating, promoted_to_benchmark=promoted, added_to_review_queue=added_to_queue)


def _answers_agree(a: str, b: str) -> bool:
    import re

    nums_a = re.findall(r"-?\d+(?:\.\d+)?", a)
    nums_b = re.findall(r"-?\d+(?:\.\d+)?", b)
    if nums_a and nums_b:
        try:
            return abs(float(nums_a[0]) - float(nums_b[0])) / max(abs(float(nums_a[0])), 1e-9) < 0.02
        except (ValueError, ZeroDivisionError):
            pass
    return a.strip().lower() == b.strip().lower()


def load_review_queue() -> list[dict]:
    if not REVIEW_QUEUE_PATH.exists():
        return []
    with open(REVIEW_QUEUE_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

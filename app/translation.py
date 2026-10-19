"""
Language handling (Section 4.1).

Detects the question's language before it reaches the Router. Supported
non-English questions are translated to English for the whole pipeline,
and the final answer (with its assumptions/scope note) is translated back.
Translation itself is done through the same model-client abstraction used
everywhere else (llm.py) rather than a separate translation API, so the
only network egress in the system stays "model API calls" -- no separate
third-party translation service is contacted.
"""
from __future__ import annotations

from langdetect import DetectorFactory, detect

from app import config
from app.llm import call_text

DetectorFactory.seed = 0  # deterministic detection


class UnsupportedLanguageError(Exception):
    pass


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"


def ensure_supported(lang: str) -> None:
    if lang not in config.SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Language '{lang}' is not in the supported set: {', '.join(sorted(config.SUPPORTED_LANGUAGES))}."
        )


def to_english(text: str, source_lang: str) -> str:
    if source_lang == "en":
        return text
    resp = call_text(
        tier="executor",
        system_prompt="Translate the user's text to English. Return ONLY the translation, no commentary.",
        user_prompt=text,
        heuristic_fallback=text,
        max_tokens=500,
    )
    return resp.text.strip() or text


def from_english(text: str, target_lang: str) -> str:
    if target_lang == "en":
        return text
    resp = call_text(
        tier="executor",
        system_prompt=f"Translate the following English text to language code '{target_lang}'. "
                      f"Return ONLY the translation, no commentary.",
        user_prompt=text,
        heuristic_fallback=text,
        max_tokens=800,
    )
    return resp.text.strip() or text

"""
PII detection and masking.

Scans column names/samples for common PII patterns (emails, phone numbers,
government-ID-shaped numbers, personal names) and masks sample values before
they are shown to any model tier. The underlying dataframe used for real
computation is never touched by this module -- only the *profile* (names,
dtypes, sample values) that gets sent to the LLM tiers is masked.
"""
from __future__ import annotations

import re

import pandas as pd

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})(?!\d)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,16}(?!\d)")
NAME_COLUMN_HINTS = {"name", "full_name", "first_name", "last_name", "customer_name", "employee_name"}
PII_COLUMN_NAME_HINTS = {
    "email", "e-mail", "phone", "mobile", "ssn", "social_security",
    "passport", "national_id", "credit_card", "card_number", "address",
} | NAME_COLUMN_HINTS


def _mask_value(value: str) -> str:
    if len(value) <= 2:
        return "*" * len(value)
    return value[0] + "*" * (len(value) - 2) + value[-1]


def column_looks_like_pii(col_name: str, sample_values: list[str]) -> bool:
    lowered = col_name.lower().strip()
    if any(hint in lowered for hint in PII_COLUMN_NAME_HINTS):
        return True
    text = " ".join(str(v) for v in sample_values)
    if EMAIL_RE.search(text) or SSN_RE.search(text) or PHONE_RE.search(text):
        return True
    return False


def mask_samples(sample_values: list[str]) -> list[str]:
    """Mask any PII-shaped substrings in a list of sample values."""
    masked: list[str] = []
    for v in sample_values:
        s = str(v)
        s = EMAIL_RE.sub(lambda m: _mask_value(m.group(0)), s)
        s = SSN_RE.sub(lambda m: _mask_value(m.group(0)), s)
        s = PHONE_RE.sub(lambda m: _mask_value(m.group(0)), s)
        masked.append(_mask_value(s) if len(s) <= 40 and s == str(v) and _looks_like_bare_name(s) else s)
    return masked


def _looks_like_bare_name(s: str) -> bool:
    parts = s.split()
    return 1 <= len(parts) <= 3 and all(p.isalpha() for p in parts) and s[:1].isupper()


def scan_dataframe(df: pd.DataFrame) -> dict[str, bool]:
    """Return {column_name: pii_detected} for a dataframe."""
    flags: dict[str, bool] = {}
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20).tolist()
        flags[col] = column_looks_like_pii(str(col), sample)
    return flags

"""
Ingestion & governance (Section 4.1).

Runs entirely locally -- no model calls. Does five things on every upload:
modality gate, profiling, PII detection, data-quality scan, multi-table join
inference. Also computes the content fingerprint used to scope caching,
sessions, and cross-session consistency.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pandas as pd

from app import config
from app.models import ColumnProfile, DatasetProfile, JoinCandidate
from app.pii import column_looks_like_pii, mask_samples


class UnsupportedFormatError(Exception):
    pass


def modality_gate(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in config.SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"'{suffix}' is not a supported tabular format. "
            f"Supported formats: {', '.join(sorted(config.SUPPORTED_FORMATS))}."
        )


def load_table(path: Path) -> pd.DataFrame:
    modality_gate(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise UnsupportedFormatError(suffix)  # pragma: no cover


def compute_fingerprint(dfs: dict[str, pd.DataFrame]) -> str:
    """Hash of actual content, stable across re-uploads of the same data."""
    h = hashlib.sha256()
    for name in sorted(dfs):
        df = dfs[name]
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:32]


def profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
    series = df[col]
    null_pct = float(series.isna().mean() * 100)
    cardinality = int(series.nunique(dropna=True))
    dtype = str(series.dtype)
    sample = series.dropna().astype(str).head(5).tolist()

    quality_flags: list[str] = []
    if null_pct > 0:
        quality_flags.append(f"nulls:{null_pct:.1f}%")
    if series.dropna().duplicated().mean() > 0.5 and cardinality > 1:
        quality_flags.append("high_duplicate_rate")
    if dtype == "object":
        numeric_coerced = pd.to_numeric(series, errors="coerce")
        if numeric_coerced.notna().mean() > 0.8 and series.notna().mean() > 0:
            quality_flags.append("type_mismatch:looks_numeric_but_stored_as_text")

    pii = column_looks_like_pii(col, sample)
    masked_sample = mask_samples(sample) if pii else sample

    return ColumnProfile(
        name=col,
        dtype=dtype,
        null_pct=round(null_pct, 2),
        cardinality=cardinality,
        sample_values=masked_sample,
        pii_detected=pii,
        quality_flags=quality_flags,
    )


def quality_summary_for(df: pd.DataFrame, columns: list[ColumnProfile]) -> list[str]:
    notes = []
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        notes.append(f"{dup_rows} fully duplicate row(s) detected")
    flagged = [c.name for c in columns if c.quality_flags]
    if flagged:
        notes.append(f"quality flags on columns: {', '.join(flagged)}")
    pii_cols = [c.name for c in columns if c.pii_detected]
    if pii_cols:
        notes.append(f"PII-shaped columns (masked for model calls): {', '.join(pii_cols)}")
    return notes


def infer_join_candidates(dfs: dict[str, pd.DataFrame]) -> list[JoinCandidate]:
    """Name-similarity + value-overlap + cardinality heuristic, then sample-validated."""
    candidates: list[JoinCandidate] = []
    names = list(dfs.keys())
    for left_name, right_name in itertools.combinations(names, 2):
        left_df, right_df = dfs[left_name], dfs[right_name]
        for lcol in left_df.columns:
            for rcol in right_df.columns:
                name_sim = _name_similarity(str(lcol), str(rcol))
                if name_sim < 0.5:
                    continue
                conf, validated = _validate_join(left_df, lcol, right_df, rcol)
                if conf <= 0:
                    continue
                candidates.append(
                    JoinCandidate(
                        left_table=left_name,
                        left_column=str(lcol),
                        right_table=right_name,
                        right_column=str(rcol),
                        confidence=round(conf, 2),
                        sample_validated=validated,
                        auto_confirmed=validated and conf >= 0.85,
                        user_confirmed=None,
                    )
                )
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _name_similarity(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    a_core = a.replace("_id", "").replace("id_", "").replace("id", "")
    b_core = b.replace("_id", "").replace("id_", "").replace("id", "")
    if a_core and a_core == b_core:
        return 0.9
    if a in b or b in a:
        return 0.6
    return 0.0


def _validate_join(left_df: pd.DataFrame, lcol: str, right_df: pd.DataFrame, rcol: str) -> tuple[float, bool]:
    try:
        left_vals = left_df[lcol].dropna()
        right_vals = right_df[rcol].dropna()
    except Exception:
        return 0.0, False
    if len(left_vals) < 5 or len(right_vals) < 5:
        return 0.3, False  # too small to sample-validate; surfaced to the user
    sample_left = left_vals.astype(str).sample(min(200, len(left_vals)), random_state=0)
    right_set = set(right_vals.astype(str).unique())
    match_rate = sample_left.isin(right_set).mean()
    if match_rate < 0.05:
        return 0.0, True
    confidence = min(0.95, 0.4 + match_rate * 0.6)
    return confidence, True


def build_dataset_profile(dfs: dict[str, pd.DataFrame]) -> DatasetProfile:
    tables: dict[str, list[ColumnProfile]] = {}
    row_counts: dict[str, int] = {}
    all_notes: list[str] = []

    for name, df in dfs.items():
        cols = [profile_column(df, c) for c in df.columns]
        tables[name] = cols
        row_counts[name] = len(df)
        notes = quality_summary_for(df, cols)
        if notes:
            all_notes.append(f"[{name}] " + "; ".join(notes))

    join_candidates = infer_join_candidates(dfs) if len(dfs) > 1 else []
    fingerprint = compute_fingerprint(dfs)

    return DatasetProfile(
        tables=tables,
        row_counts=row_counts,
        join_candidates=join_candidates,
        quality_summary="\n".join(all_notes) if all_notes else "No data-quality issues detected.",
        content_fingerprint=fingerprint,
    )

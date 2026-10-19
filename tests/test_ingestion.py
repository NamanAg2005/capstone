from pathlib import Path

import pandas as pd
import pytest

from app import ingestion

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_data"


def test_modality_gate_rejects_unsupported_extension(tmp_path):
    bad_file = tmp_path / "scan.pdf"
    bad_file.write_bytes(b"%PDF-1.4")
    with pytest.raises(ingestion.UnsupportedFormatError):
        ingestion.load_table(bad_file)


def test_load_sample_orders():
    df = ingestion.load_table(SAMPLE_DIR / "orders.csv")
    assert len(df) == 35
    assert "quantity" in df.columns


def test_profile_detects_nulls_and_pii():
    dfs = {
        "orders": ingestion.load_table(SAMPLE_DIR / "orders.csv"),
        "customers": ingestion.load_table(SAMPLE_DIR / "customers.csv"),
    }
    profile = ingestion.build_dataset_profile(dfs)
    order_cols = {c.name: c for c in profile.tables["orders"]}
    assert order_cols["discount_pct"].null_pct > 0
    assert "nulls" in " ".join(order_cols["discount_pct"].quality_flags)

    customer_cols = {c.name: c for c in profile.tables["customers"]}
    assert customer_cols["email"].pii_detected
    assert all("@" not in v or "*" in v for v in customer_cols["email"].sample_values)


def test_join_inference_finds_customer_id():
    dfs = {
        "orders": ingestion.load_table(SAMPLE_DIR / "orders.csv"),
        "customers": ingestion.load_table(SAMPLE_DIR / "customers.csv"),
    }
    profile = ingestion.build_dataset_profile(dfs)
    pairs = {(j.left_column, j.right_column) for j in profile.join_candidates}
    assert ("customer_id", "customer_id") in pairs


def test_fingerprint_stable_and_changes_with_content():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})
    df3 = pd.DataFrame({"a": [1, 2, 4]})
    assert ingestion.compute_fingerprint({"t": df1}) == ingestion.compute_fingerprint({"t": df2})
    assert ingestion.compute_fingerprint({"t": df1}) != ingestion.compute_fingerprint({"t": df3})

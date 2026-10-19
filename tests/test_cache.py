from app.cache import ExactCache
from app.vectorstore import VectorStore, get_store
import app.cache as cache_module


def test_exact_cache_roundtrip(tmp_path, monkeypatch):
    # Point the module-level vector store (used by ExactCache.put for the
    # semantic index) at a throwaway file for this test.
    monkeypatch.setattr(cache_module, "get_store", lambda: VectorStore(tmp_path / "vec.sqlite3"))

    c = ExactCache(tmp_path / "cache.sqlite3")
    assert c.get("fp1", "count rows", "pandas") is None

    c.put("fp1", "count rows", "pandas", code="result = len(df)", result={"result_repr": "35"})
    hit = c.get("fp1", "count rows", "pandas")
    assert hit is not None
    assert hit["code"] == "result = len(df)"
    assert hit["result"]["result_repr"] == "35"


def test_exact_cache_misses_on_different_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "get_store", lambda: VectorStore(tmp_path / "vec.sqlite3"))
    c = ExactCache(tmp_path / "cache.sqlite3")
    c.put("fp1", "count rows", "pandas", code="result = len(df)", result={"result_repr": "35"})
    assert c.get("fp2", "count rows", "pandas") is None


def test_semantic_lookup_finds_similar_description(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "get_store", lambda: VectorStore(tmp_path / "vec.sqlite3"))
    c = ExactCache(tmp_path / "cache.sqlite3")
    c.put("fp1", "total number of orders", "pandas", code="result = len(orders)", result={"result_repr": "35"})
    hit = c.semantic_lookup("fp1", "what is the total number of orders", "pandas", min_similarity=0.5)
    assert hit is not None
    assert hit["code"] == "result = len(orders)"

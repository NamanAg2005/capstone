from app.vectorstore import VectorStore


def test_add_and_query(tmp_path):
    store = VectorStore(tmp_path / "vec.sqlite3")
    store.add("schema_index", "col1", "customer email address", {"table": "customers"})
    store.add("schema_index", "col2", "order quantity integer", {"table": "orders"})

    hits = store.query("schema_index", "email contact info", top_k=1)
    assert hits[0]["doc_id"] == "col1"


def test_metadata_filter_scopes_results(tmp_path):
    store = VectorStore(tmp_path / "vec.sqlite3")
    store.add("session_memory", "a", "total revenue by quarter", {"session_id": "s1"})
    store.add("session_memory", "b", "total revenue by quarter", {"session_id": "s2"})

    hits = store.query("session_memory", "revenue by quarter", top_k=5, metadata_filter={"session_id": "s1"})
    assert len(hits) == 1
    assert hits[0]["doc_id"] == "a"


def test_replace_same_doc_id(tmp_path):
    store = VectorStore(tmp_path / "vec.sqlite3")
    store.add("c", "x", "first version", {})
    store.add("c", "x", "second version", {})
    hits = store.query("c", "second version", top_k=5)
    assert len(hits) == 1
    assert hits[0]["text"] == "second version"

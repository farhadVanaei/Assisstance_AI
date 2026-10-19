"""Unit tests for SQLite persistence."""

from __future__ import annotations


def test_add_and_list_sources(tmp_data):
    from ollama_rag.store import SQLiteStore

    store = SQLiteStore(tmp_data / "a.db")
    src = store.add_source("doc.txt", "text", "این یک متن آزمایشی بلند برای ذخیره است." * 2)
    assert src is not None
    chunks = store.add_chunks(src.id, src.name, "text", ["قطعه یک آزمایشی کافی طولانی است.", "قطعه دو آزمایشی کافی طولانی است."])
    assert len(chunks) == 2
    listed = store.list_sources()
    assert len(listed) == 1
    assert listed[0].chunk_count == 2
    assert store.chunk_count() == 2


def test_dedup_on_source_hash(tmp_data):
    from ollama_rag.store import SQLiteStore

    store = SQLiteStore(tmp_data / "b.db")
    text = "محتوای یکسان برای بررسی تکراری بودن منبع در پایگاه."
    assert store.add_source("a", "text", text) is not None
    assert store.add_source("b", "text", text) is None


def test_chat_history(tmp_data):
    from ollama_rag.store import SQLiteStore

    store = SQLiteStore(tmp_data / "c.db")
    store.append_history("s1", "user", "سلام")
    store.append_history("s1", "assistant", "درود")
    hist = store.get_history("s1")
    assert len(hist) == 2
    assert hist[0]["role"] == "user"


def test_delete_source(tmp_data):
    from ollama_rag.store import SQLiteStore

    store = SQLiteStore(tmp_data / "d.db")
    text = "متن قابل حذف برای آزمایش حذف منبع از پایگاه دانش."
    src = store.add_source("x", "text", text)
    store.add_chunks(src.id, "x", "text", [text])
    deleted = store.delete_source(src.id)
    assert deleted == 1
    assert store.chunk_count() == 0

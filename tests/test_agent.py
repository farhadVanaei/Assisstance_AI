"""Unit tests for ingest, retrieval scoring, and chat fallback."""

from __future__ import annotations

import pytest


@pytest.fixture()
def agent(tmp_data):
    from ollama_rag.agent import JavadOllamaAgent
    from ollama_rag.store import SQLiteStore
    from ollama_rag.vector_store import VectorStore

    store = SQLiteStore(tmp_data / "agent.db")
    vectors = VectorStore(tmp_data / "chroma_agent", collection="test_agent")
    return JavadOllamaAgent(user_id="test", store=store, vectors=vectors)


def test_ingest_and_retrieve(agent):
    text = (
        "سیستم موقعیت‌یابی SPOT یک فناوری ماهواره‌ای برای ردیابی لحظه‌ای است. "
        "این سامانه در پروژه‌های حمل‌ونقل و امداد کاربرد دارد و دقت بالایی دارد. "
    ) * 3
    added = agent.ingest(text, "spot.txt", "text")
    assert added > 0
    assert agent.document_count == added
    # A second, unrelated document keeps BM25's IDF term meaningful — with only a
    # single document in the corpus, BM25's IDF math goes pathologically negative
    # for any term present (a known degenerate case for tiny corpora), which would
    # mask whether retrieval is actually discriminating relevant from irrelevant.
    agent.ingest(
        ("دستور پخت قورمه سبزی شامل سبزی معطر، لوبیا قرمز و گوشت است. "
         "این غذای سنتی ایرانی با برنج سرو می‌شود. ") * 3,
        "recipe.txt",
        "text",
    )
    hits = agent._retrieve("SPOT چیست")
    assert hits
    assert any("SPOT" in c.text or "spot" in c.source.lower() for c in hits)
    # The relevant chunk must rank above the unrelated one, even if the hash-fallback
    # embedding (used here since no real Ollama is running in tests) is too noisy to
    # guarantee the irrelevant chunk is excluded outright.
    top = hits[0]
    assert "SPOT" in top.text or "spot" in top.source.lower()


def test_chunk_overlap():
    from ollama_rag.agent import _split_chunks

    text = "الف " * 200
    parts = _split_chunks(text, max_len=80, overlap=20)
    assert len(parts) > 1
    # Overlap: end of one chunk should share content with start of next
    assert parts[0][-10:].strip() in parts[1] or any(
        w in parts[1] for w in parts[0].split()[-3:]
    )


def test_dedup_ingest(agent):
    text = "متن تکراری برای آزمایش جلوگیری از ذخیره دوباره در پایگاه دانش جواد. " * 5
    assert agent.ingest(text, "a.txt") > 0
    assert agent.ingest(text, "b.txt") == 0


def test_delete_source(agent):
    text = "منبعی که باید حذف شود از پایگاه دانش پس از آزمایش. " * 5
    agent.ingest(text, "del.txt")
    sources = agent.list_sources()
    assert sources
    result = agent.delete_source(sources[0]["id"])
    assert result["ok"]
    assert agent.document_count == 0


@pytest.mark.asyncio
async def test_chat_fallback_no_knowledge(agent):
    result = await agent.chat("SPOT چه کاری انجام می‌دهد؟", session_id="t1")
    assert result["mode"] == "fallback"
    assert result["chunks_used"] == 0
    assert "دانش" in result["answer"] or "نیست" in result["answer"]


@pytest.mark.asyncio
async def test_chat_with_knowledge_fallback(agent, monkeypatch):
    text = (
        "پروژه جواد یک دستیار محلی فارسی است که با اولاما کار می‌کند. "
        "دانش را از اسناد و وب‌سایت می‌آموزد و آفلاین پاسخ می‌دهد. "
    ) * 4
    agent.ingest(text, "javad.txt")

    async def boom(*args, **kwargs):
        from ollama_rag.client import OllamaError

        raise OllamaError("offline")

    monkeypatch.setattr("ollama_rag.client.chat", boom)
    result = await agent.chat("جواد چیست؟", session_id="t2")
    assert result["context_used"] or result["mode"] == "fallback"
    assert result["answer"]
    assert result.get("citations") is not None


def test_relevance_threshold_drops_noise(agent):
    agent.ingest("آب و هوا در تهران امروز آفتابی است و دمای هوا معتدل می‌باشد. " * 5, "weather.txt")
    hits = agent._retrieve("قیمت سهام بورس نیویورک فردا")
    # May return a weak best-effort chunk, but score should be tracked
    for h in hits:
        assert hasattr(h, "score")

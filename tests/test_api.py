"""Integration tests for main API routes."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_data, monkeypatch):
    # Ensure main picks up reloaded agent/store
    import importlib
    import main as main_mod

    importlib.reload(main_mod)
    return TestClient(main_mod.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "chunks" in data
    assert "vector_store" in data
    assert "embed_model" in data
    assert "metrics" in data


def test_upload_and_list_and_delete(client):
    content = ("دانش آزمایشی جواد درباره سامانه موقعیت‌یابی SPOT است. " * 10).encode("utf-8")
    r = client.post(
        "/api/rag/upload",
        files=[("files", ("spot.txt", io.BytesIO(content), "text/plain"))],
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and body
    assert body[0]["chunks_added"] > 0

    listed = client.get("/api/knowledge")
    assert listed.status_code == 200
    sources = listed.json()["sources"]
    assert sources
    sid = sources[0]["id"]

    deleted = client.delete(f"/api/rag/source/{sid}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True

    listed2 = client.get("/api/knowledge")
    assert listed2.json()["count"] == 0


def test_chat_empty_knowledge(client):
    r = client.post("/api/ollama/chat", json={"prompt": "SPOT چیست؟", "session_id": "api-t"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "citations" in data


def test_chat_stream(client):
    r = client.post(
        "/api/ollama/chat/stream",
        json={"prompt": "سلام", "session_id": "api-stream"},
    )
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines
    import json

    events = [json.loads(ln) for ln in lines]
    assert any(e.get("type") == "done" for e in events)


def test_memory_clear(client):
    r = client.post("/api/memory/clear")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_website_scrape_bad_url(client):
    # Too-short / invalid scheme → validation or scraper ValueError
    r = client.post("/api/website/scrape", json={"url": "notaurl"})
    assert r.status_code in {400, 422}
    r2 = client.post("/api/website/scrape", json={"url": "ftp://example.com/page"})
    assert r2.status_code == 400

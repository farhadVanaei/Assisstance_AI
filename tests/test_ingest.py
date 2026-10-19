"""Unit tests for document ingestion."""

from __future__ import annotations

import io
import json

import pytest


def test_validate_rejects_unsupported():
    from rag.ingest import IngestError, validate_upload

    with pytest.raises(IngestError):
        validate_upload("virus.exe", b"MZ\x00\x00" + b"x" * 100)


def test_validate_rejects_oversized(monkeypatch):
    from rag.ingest import IngestError, validate_upload
    import ollama_rag.config as cfg

    monkeypatch.setattr(cfg, "MAX_UPLOAD_BYTES", 100)
    # Re-import bound constant used inside ingest — patch module attribute
    import rag.ingest as ingest_mod

    monkeypatch.setattr(ingest_mod, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(IngestError):
        validate_upload("big.txt", b"a" * 200)


@pytest.mark.asyncio
async def test_ingest_txt():
    from rag.ingest import ingest_file

    data = "سلام دنیا. این یک سند متنی آزمایشی است.".encode("utf-8")
    result = await ingest_file("note.txt", data)
    assert result["kind"] == "text"
    assert "سلام" in result["text"]


@pytest.mark.asyncio
async def test_ingest_csv():
    from rag.ingest import ingest_file

    data = "name,city\nAli,Tehran\nSara,Isfahan\n".encode("utf-8")
    result = await ingest_file("people.csv", data)
    assert result["kind"] == "csv"
    assert "Tehran" in result["text"]


@pytest.mark.asyncio
async def test_ingest_json():
    from rag.ingest import ingest_file

    payload = json.dumps(
        [{"title": "SPOT", "desc": "سامانه موقعیت‌یابی"}, {"title": "JAVAD", "desc": "دستیار"}],
        ensure_ascii=False,
    ).encode("utf-8")
    result = await ingest_file("data.json", payload)
    assert result["kind"] == "json"
    assert "SPOT" in result["text"]


@pytest.mark.asyncio
async def test_ingest_docx():
    pytest.importorskip("docx")
    from docx import Document
    from rag.ingest import ingest_file

    doc = Document()
    doc.add_paragraph("این یک سند ورد آزمایشی برای جواد است.")
    buf = io.BytesIO()
    doc.save(buf)
    result = await ingest_file("t.docx", buf.getvalue())
    assert result["kind"] == "docx"
    assert "ورد" in result["text"]


@pytest.mark.asyncio
async def test_binary_as_text_rejected():
    from rag.ingest import IngestError, ingest_file

    with pytest.raises(IngestError):
        await ingest_file("x.bin", b"\x00\x01\x02\x03" * 50)

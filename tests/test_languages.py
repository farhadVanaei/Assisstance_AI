"""Multi-language prompt / detection tests. Persian remains default."""

from __future__ import annotations

from ollama_rag.languages import (
    detect_language,
    normalize_lang,
    system_prompt,
    chitchat_cues,
    leak_markers,
)


def test_default_is_persian():
    assert normalize_lang(None) == "fa"
    assert "فارسی" in system_prompt("fa") or "جواد" in system_prompt("fa")


def test_detect_persian():
    assert detect_language("سلام، حالت چطوره؟ من یک سؤال دارم.") == "fa"


def test_detect_english():
    assert detect_language("Hello, how are you doing today my friend?") == "en"


def test_chitchat_and_leaks_per_language():
    assert "سلام" in chitchat_cues("fa")
    assert "hello" in chitchat_cues("en")
    assert any("system:" in m.lower() or m == "system:" for m in leak_markers("fa"))
    assert len(leak_markers("en")) >= 3

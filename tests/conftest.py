"""Pytest fixtures — isolated temp data dirs for each test session."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))


@pytest.fixture()
def tmp_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("JAVAD_DATA_DIR", str(data))
    monkeypatch.setenv("JAVAD_SQLITE_PATH", str(data / "test.db"))
    monkeypatch.setenv("JAVAD_CHROMA_PATH", str(data / "chroma"))
    monkeypatch.setenv("JAVAD_API_KEY", "")
    monkeypatch.setenv("JAVAD_AUTH_REQUIRED", "false")
    monkeypatch.setenv("JAVAD_CORS_ALLOW_ALL", "true")
    # Reload config-dependent modules cleanly
    import importlib

    import ollama_rag.config as cfg
    importlib.reload(cfg)
    import ollama_rag.store as store_mod
    importlib.reload(store_mod)
    store_mod._store = None
    import ollama_rag.vector_store as vs_mod
    importlib.reload(vs_mod)
    vs_mod._vs = None
    import ollama_rag.agent as agent_mod
    importlib.reload(agent_mod)
    agent_mod.reset_agents()
    return data

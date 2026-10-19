"""Ollama HTTP client (fully local — default http://127.0.0.1:11434)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("JAVAD_OLLAMA_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("JAVAD_OLLAMA_EMBED_MODEL", "")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error (Persian message)."""


def _resolve_model(requested: Optional[str], available: List[str]) -> str:
    name = (requested or CHAT_MODEL).strip()
    if not available:
        return name
    if name in available:
        return name
    # Prefer exact :latest variant
    for cand in (f"{name}:latest", name.split(":")[0], f"{name.split(':')[0]}:latest"):
        if cand in available:
            return cand
    # Fallback to first installed model
    return available[0]


async def check_ollama() -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            return {"ok": True, "models": models, "base": OLLAMA_BASE}
    except Exception as exc:
        raise OllamaError(
            "اتصال به اولاما برقرار نشد. لطفاً سرویس محلی اولاما را روشن کنید "
            "و مدل را دریافت کنید."
        ) from exc


async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    info = await check_ollama()
    resolved = _resolve_model(model or CHAT_MODEL, info.get("models") or [])
    payload = {
        "model": resolved,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            if r.status_code == 404:
                raise OllamaError(
                    "مدل محلی پیدا نشد. لطفاً مدل را با دستور دریافت مدل نصب کنید."
                )
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content") or ""
            return str(content).strip()
    except OllamaError:
        raise
    except httpx.ConnectError as exc:
        raise OllamaError(
            "اولاما در دسترس نیست. سرویس محلی را روشن کنید و دوباره تلاش کنید."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError("خطا در گفتگو با مدل محلی رخ داد.") from exc


async def embed(texts: List[str], model: Optional[str] = None) -> Optional[List[List[float]]]:
    m = model or EMBED_MODEL
    if not m or not texts:
        return None
    vectors: List[List[float]] = []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": m, "prompt": text},
                )
                if r.status_code >= 400:
                    return None
                vectors.append(list(r.json().get("embedding") or []))
        return vectors if all(vectors) else None
    except Exception:
        return None

"""Ollama HTTP client (fully local — default http://127.0.0.1:11434)."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ollama_rag.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_HOST

logger = logging.getLogger("javad.ollama")

OLLAMA_BASE = OLLAMA_HOST


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error (Persian message)."""


def _resolve_model(requested: Optional[str], available: List[str]) -> str:
    name = (requested or CHAT_MODEL).strip()
    if not available:
        return name
    if name in available:
        return name
    for cand in (f"{name}:latest", name.split(":")[0], f"{name.split(':')[0]}:latest"):
        if cand in available:
            return cand
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


async def check_embed_model(model: Optional[str] = None) -> Dict[str, Any]:
    """Check whether the configured embedding model is available."""
    m = model or EMBED_MODEL
    try:
        info = await check_ollama()
        models = info.get("models") or []
        names = {x.split(":")[0] for x in models} | set(models)
        ok = m in models or m in names or f"{m}:latest" in models
        return {"ok": ok, "model": m, "available_models": models}
    except OllamaError:
        return {"ok": False, "model": m, "available_models": []}


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


async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> AsyncIterator[str]:
    """Yield tokens from Ollama streaming chat."""
    info = await check_ollama()
    resolved = _resolve_model(model or CHAT_MODEL, info.get("models") or [])
    payload = {
        "model": resolved,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 256,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as r:
                if r.status_code == 404:
                    raise OllamaError(
                        "مدل محلی پیدا نشد. لطفاً مدل را با دستور دریافت مدل نصب کنید."
                    )
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (data.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if data.get("done"):
                        break
    except OllamaError:
        raise
    except httpx.ConnectError as exc:
        raise OllamaError(
            "اولاما در دسترس نیست. سرویس محلی را روشن کنید و دوباره تلاش کنید."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError("خطا در گفتگو با مدل محلی رخ داد.") from exc


async def embed(texts: List[str], model: Optional[str] = None) -> Optional[List[List[float]]]:
    """Embed texts via Ollama. Tries /api/embed then /api/embeddings."""
    m = model or EMBED_MODEL
    if not m or not texts:
        return None
    vectors: List[List[float]] = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Prefer batch /api/embed (newer Ollama)
            try:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embed",
                    json={"model": m, "input": texts},
                )
                if r.status_code < 400:
                    data = r.json()
                    batch = data.get("embeddings") or []
                    if batch and len(batch) == len(texts):
                        return [list(v) for v in batch]
            except Exception:
                pass

            for text in texts:
                r = await client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": m, "prompt": text},
                )
                if r.status_code >= 400:
                    logger.warning("embed HTTP %s for model %s", r.status_code, m)
                    return None
                vectors.append(list(r.json().get("embedding") or []))
        return vectors if all(vectors) else None
    except Exception as exc:
        logger.warning("embed failed: %s", exc)
        return None


def embed_sync(texts: List[str], model: Optional[str] = None) -> Optional[List[List[float]]]:
    """Synchronous embedding for ingest paths that run outside async context."""
    m = model or EMBED_MODEL
    if not m or not texts:
        return None
    vectors: List[List[float]] = []
    try:
        with httpx.Client(timeout=120.0) as client:
            try:
                r = client.post(
                    f"{OLLAMA_BASE}/api/embed",
                    json={"model": m, "input": texts},
                )
                if r.status_code < 400:
                    data = r.json()
                    batch = data.get("embeddings") or []
                    if batch and len(batch) == len(texts):
                        return [list(v) for v in batch]
            except Exception:
                pass
            for text in texts:
                r = client.post(
                    f"{OLLAMA_BASE}/api/embeddings",
                    json={"model": m, "prompt": text},
                )
                if r.status_code >= 400:
                    return None
                vectors.append(list(r.json().get("embedding") or []))
        return vectors if all(vectors) else None
    except Exception:
        return None

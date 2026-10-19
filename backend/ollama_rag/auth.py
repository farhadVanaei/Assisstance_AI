"""Optional API-key authentication and per-user scoping."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from fastapi import Header, HTTPException, Request

from ollama_rag.config import API_KEY, AUTH_REQUIRED

logger = logging.getLogger("javad.auth")


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def resolve_user_id(
    request: Request,
    x_api_key: Optional[str] = None,
    x_user_id: Optional[str] = None,
) -> str:
    """
    Resolve authenticated user id.
    - If JAVAD_API_KEY is set (or AUTH_REQUIRED), require matching X-API-Key.
    - Optional X-User-Id scopes knowledge; defaults to hash of API key or 'default'.
    """
    provided = (x_api_key or "").strip()
    header_user = (x_user_id or "").strip()

    if API_KEY:
        if not provided or provided != API_KEY:
            raise HTTPException(status_code=401, detail="کلید API نامعتبر یا موجود نیست.")
        return header_user or f"user_{_hash_key(provided)}"

    if AUTH_REQUIRED:
        raise HTTPException(
            status_code=401,
            detail="احراز هویت فعال است ولی JAVAD_API_KEY تنظیم نشده.",
        )

    # Open local mode
    return header_user or "default"


async def require_user(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> str:
    return resolve_user_id(request, x_api_key=x_api_key, x_user_id=x_user_id)

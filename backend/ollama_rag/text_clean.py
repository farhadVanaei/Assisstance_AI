"""Sanitize scraped/document text before RAG + Ollama (no chrome, URLs, or junk)."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI = re.compile(r"\s+")

# ناوبری / قالب‌های تکراری وب (فارسی و باقیمانده‌های رایج)
_BOILER = [
    r"پرش به محتوا",
    r"پرش به منو",
    r"پرش به اصلی",
    r"پرش به ناوبری",
    r"رفتن به محتوا",
    r"رفتن به منو",
    r"منوی اصلی",
    r"فهرست مطالب",
    r"فهرست اصلی",
    r"جستجو\s*کنید",
    r"جست‌?جو",
    r"ورود\s*/\s*ثبت[\s‌]*نام",
    r"عضویت\s*/\s*ورود",
    r"حقوق محفوظ است",
    r"تمامی حقوق",
    r"کلیه حقوق",
    r"خانه\s*[|>›»]\s*",
    r"skip to (main )?content",
    r"skip to navigation",
    r"main menu",
    r"cookie(s)? (policy|settings|consent|banner)",
    r"accept (all )?cookies",
    r"privacy policy",
    r"terms of (use|service)",
    r"all rights reserved",
    r"©\s*\d{4}",
    r"منبع\s*:",
    r"source\s*:",
    r"share on",
    r"follow us",
    r"powered by",
    r"read more",
    r"click here",
]


def clean_text(text: str) -> str:
    """General cleanup for storage and display."""
    return sanitize_document(text)


def sanitize_document(text: str) -> str:
    """
    Strip HTML scraps, URLs, emails, and website chrome so Ollama
    only receives readable content.
    """
    if not text:
        return ""
    text = str(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    for pat in _BOILER:
        text = re.sub(pat, " ", text, flags=re.I)
    # Arabic Yeh/Kaf → Persian
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    # Drop leftover script/style blocks if any escaped
    text = re.sub(r"(?is)script|style|noscript", " ", text)
    text = _MULTI.sub(" ", text).strip(" -\u2013\u2014|·•\t")
    return text


def sanitize_for_prompt(text: str, max_chars: int = 3500) -> str:
    """Final pass before injecting context into the Ollama system message."""
    text = sanitize_document(text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip() + "…"
    return text

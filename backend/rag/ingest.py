"""Extract text from PDFs, HTML, images, and remote URLs for RAG ingest."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader


async def ingest_file(filename: str, data: bytes) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(data)
        kind = "pdf"
    elif suffix in {".html", ".htm"}:
        text = _extract_html(data.decode("utf-8", errors="ignore"))
        kind = "html"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        text = _extract_image_meta(filename, data)
        kind = "image"
    elif suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
        kind = "text"
    else:
        text = data.decode("utf-8", errors="ignore")
        kind = "binary-as-text"

    return {
        "source": filename,
        "kind": kind,
        "text": text,
        "chars": len(text),
    }


async def ingest_url(url: str, timeout: float = 20.0) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("فقط آدرس‌های http و https پشتیبانی می‌شوند.")

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": "JAVAD-Learning-Hub/1.0"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        raw = response.content

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        text = _extract_pdf(raw)
        kind = "pdf-url"
    else:
        encoding = response.encoding or "utf-8"
        html = raw.decode(encoding, errors="ignore")
        text = _extract_html(html)
        kind = "web"

    source = url
    return {"source": source, "kind": kind, "text": text, "chars": len(text)}


def _extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _extract_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    body = soup.get_text(separator=" ", strip=True)
    if title:
        return f"{title}\n{body}"
    return body


def _extract_image_meta(filename: str, data: bytes) -> str:
    """Lightweight image ingest without OCR dependency.

    Stores filename, dimensions, and a Persian note so screenshots
    remain discoverable in the knowledge base.
    """
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        width, height = image.size
        mode = image.mode
        fmt = image.format or Path(filename).suffix.lstrip(".").upper()
        return (
            f"تصویر بارگذاری‌شده: {filename}. "
            f"ابعاد: {width}×{height} پیکسل، حالت رنگ: {mode}، قالب: {fmt}. "
            "این اسکرین‌شات به‌عنوان مدرک بصری در پایگاه دانش جواد ثبت شد. "
            "برای استخراج متن دقیق، لطفاً نسخه متنی یا PDF همان محتوا را نیز آپلود کنید."
        )
    except Exception:
        return (
            f"تصویر بارگذاری‌شده: {filename} ({len(data)} بایت). "
            "محتوای بصری در پایگاه دانش جواد ثبت شد."
        )

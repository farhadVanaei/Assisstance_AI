"""Extract text from PDFs, Office docs, HTML, images (OCR), CSV/JSON, and audio."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from ollama_rag.config import MEDIA_DIR

logger = logging.getLogger("javad.ingest")

# Skip tiny embedded images (icons, bullets, tracking pixels) — not worth surfacing.
_MIN_IMAGE_DIM = 80


def _save_media(data: bytes, ext: str) -> Optional[str]:
    """Save extracted image bytes under MEDIA_DIR, content-hashed for dedup.
    Returns a URL path (e.g. "/media/<hash>.png") or None on failure."""
    if not data:
        return None
    ext = (ext or "png").lstrip(".").lower() or "png"
    digest = hashlib.sha256(data).hexdigest()[:24]
    filename = f"{digest}.{ext}"
    try:
        path = MEDIA_DIR / filename
        if not path.exists():
            path.write_bytes(data)
        return f"/media/{filename}"
    except Exception as exc:
        logger.info("Failed to save extracted image: %s", exc)
        return None

SUPPORTED_SUFFIXES: Set[str] = {
    ".pdf",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".json",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
}


class IngestError(ValueError):
    """Raised when a file cannot be ingested cleanly."""


def validate_upload(filename: str, data: bytes) -> str:
    """Validate size/type; return normalized suffix. Raises IngestError."""
    if not filename:
        raise IngestError("نام فایل نامعتبر است.")
    if not data:
        raise IngestError("فایل خالی است.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestError(
            f"نوع فایل «{suffix or 'نامشخص'}» پشتیبانی نمی‌شود. "
            f"فرمت‌های مجاز: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    # Reject obvious binary masquerading as text extensions
    if suffix in {".txt", ".md", ".csv", ".json", ".html", ".htm"}:
        # Allow UTF-8/latin with high null ratio check
        nulls = data[:4096].count(b"\x00")
        if nulls > 8:
            raise IngestError("فایل باینری به‌عنوان متن پذیرفته نمی‌شود.")
    return suffix


async def ingest_file(filename: str, data: bytes) -> dict:
    suffix = validate_upload(filename, data)
    images: List[str] = []
    sections: List[Dict[str, Any]] = []

    if suffix == ".pdf":
        sections = _extract_pdf_sections(data)
        if sections:
            text = "\n".join(s["text"] for s in sections if s.get("text")).strip()
        else:
            text = _extract_pdf(data)
        kind = "pdf"
    elif suffix in {".html", ".htm"}:
        text = _extract_html(data.decode("utf-8", errors="ignore"))
        kind = "html"
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        text = _extract_image_ocr(filename, data)
        kind = "image"
        saved = _save_media(data, Path(filename).suffix)
        if saved:
            images = [saved]
    elif suffix in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
        kind = "text"
    elif suffix == ".docx":
        sections = _extract_docx_sections(data)
        if sections:
            text = "\n".join(s["text"] for s in sections if s.get("text")).strip()
        else:
            text = _extract_docx(data)
        kind = "docx"
    elif suffix == ".pptx":
        text = _extract_pptx(data)
        kind = "pptx"
    elif suffix == ".xlsx":
        text = _extract_xlsx(data)
        kind = "xlsx"
    elif suffix == ".csv":
        text = _extract_csv(data)
        kind = "csv"
    elif suffix == ".json":
        text = _extract_json(data)
        kind = "json"
    elif suffix in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        text = await _extract_audio(filename, data)
        kind = "audio"
    else:
        raise IngestError(f"نوع فایل پشتیبانی نمی‌شود: {suffix}")

    if sections:
        for s in sections:
            for im in s.get("images") or []:
                if im not in images:
                    images.append(im)

    return {
        "source": filename,
        "kind": kind,
        "text": text,
        "chars": len(text),
        "images": images,
        "sections": sections,
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

    return {"source": url, "kind": kind, "text": text, "chars": len(text)}


def _extract_pdf(data: bytes) -> str:
    sections = _extract_pdf_sections(data)
    if sections:
        return "\n".join(s["text"] for s in sections if s.get("text")).strip()

    # pypdf frequently drops word spacing and/or mis-orders RTL (Persian/Arabic)
    # text, corrupting tokenization-based retrieval; PyMuPDF handles both correctly.
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _extract_pdf_sections(
    data: bytes, max_images_per_page: int = 10
) -> List[Dict[str, Any]]:
    """Per-page text + images, so each page's images stay attached to that page's
    own text instead of leaking onto chunks from unrelated pages."""
    try:
        import pymupdf
    except Exception as exc:
        logger.info("pymupdf unavailable for PDF extraction: %s", exc)
        return []

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            raw_pages: List[tuple] = []  # (page_text, [(hash, img_bytes, ext)])
            hash_page_count: Dict[str, int] = {}
            for page in doc:
                page_text = page.get_text() or ""
                candidates = []
                seen_on_page: Set[str] = set()
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        info = doc.extract_image(xref)
                    except Exception:
                        continue
                    img_bytes = info.get("image")
                    width = info.get("width") or 0
                    height = info.get("height") or 0
                    if not img_bytes or width < _MIN_IMAGE_DIM or height < _MIN_IMAGE_DIM:
                        continue
                    h = hashlib.sha256(img_bytes).hexdigest()
                    if h in seen_on_page:
                        continue
                    seen_on_page.add(h)
                    candidates.append((h, img_bytes, info.get("ext") or "png"))
                    hash_page_count[h] = hash_page_count.get(h, 0) + 1
                raw_pages.append((page_text, candidates))
    except Exception as exc:
        logger.info("PDF section extraction failed: %s", exc)
        return []

    # An image repeated across many pages is a header/footer logo or watermark,
    # not page content — attaching it to every page's chunks is as misleading as
    # attaching the whole document's images to every chunk.
    repeat_threshold = max(3, page_count // 3) if page_count > 1 else 1
    boilerplate = {
        h for h, n in hash_page_count.items() if page_count > 1 and n >= repeat_threshold
    }

    sections: List[Dict[str, Any]] = []
    for page_text, candidates in raw_pages:
        page_images: List[str] = []
        for h, img_bytes, ext in candidates:
            if h in boilerplate or len(page_images) >= max_images_per_page:
                continue
            saved = _save_media(img_bytes, ext)
            if saved and saved not in page_images:
                page_images.append(saved)
        sections.append({"text": page_text, "images": page_images})
    return sections


def _extract_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title else ""
    body = soup.get_text(separator=" ", strip=True)
    if title:
        return f"{title}\n{body}"
    return body


def _extract_image_ocr(filename: str, data: bytes) -> str:
    """OCR via pytesseract when available; otherwise metadata + note."""
    meta = _image_meta(filename, data)
    ocr_text = ""
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(data))
        # Persian + English
        ocr_text = pytesseract.image_to_string(image, lang="fas+eng") or ""
        ocr_text = ocr_text.strip()
    except Exception as exc:
        logger.info("OCR unavailable for %s: %s", filename, exc)
        try:
            import easyocr

            reader = easyocr.Reader(["fa", "en"], gpu=False, verbose=False)
            from PIL import Image
            import numpy as np

            image = Image.open(io.BytesIO(data)).convert("RGB")
            result = reader.readtext(np.array(image), detail=0)
            ocr_text = "\n".join(result).strip()
        except Exception as exc2:
            logger.info("easyocr also unavailable: %s", exc2)

    if ocr_text and len(ocr_text) >= 8:
        return f"{meta}\n\nمتن استخراج‌شده از تصویر:\n{ocr_text}"
    return (
        f"{meta}\n"
        "برای استخراج متن دقیق‌تر، موتور OCR (tesseract یا easyocr) را نصب کنید "
        "یا نسخه متنی/PDF همان محتوا را آپلود کنید."
    )


def _image_meta(filename: str, data: bytes) -> str:
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        width, height = image.size
        mode = image.mode
        fmt = image.format or Path(filename).suffix.lstrip(".").upper()
        return (
            f"تصویر بارگذاری‌شده: {filename}. "
            f"ابعاد: {width}×{height} پیکسل، حالت رنگ: {mode}، قالب: {fmt}."
        )
    except Exception:
        return f"تصویر بارگذاری‌شده: {filename} ({len(data)} بایت)."


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise IngestError("بسته python-docx نصب نیست.") from exc
    doc = Document(io.BytesIO(data))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def _extract_docx_images(data: bytes, max_images: int = 10) -> List[str]:
    """Extract embedded images from a .docx via its package relationships."""
    try:
        from docx import Document
    except Exception as exc:
        logger.info("python-docx unavailable for image extraction: %s", exc)
        return []

    urls: List[str] = []
    try:
        doc = Document(io.BytesIO(data))
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            if len(urls) >= max_images:
                break
            try:
                img_bytes = rel.target_part.blob
                ext = Path(rel.target_part.partname).suffix or ".png"
            except Exception:
                continue
            if len(img_bytes) < 1024:  # skip tiny decorative images
                continue
            saved = _save_media(img_bytes, ext)
            if saved and saved not in urls:
                urls.append(saved)
    except Exception as exc:
        logger.info("DOCX image extraction failed: %s", exc)
    return urls


_DOCX_RELS_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_DOCX_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _extract_docx_sections(data: bytes, max_images: int = 10) -> List[Dict[str, Any]]:
    """Per-paragraph-run text + images, so each image stays attached to the
    paragraph it appears next to instead of being dumped on every chunk."""
    try:
        from docx import Document
    except Exception as exc:
        logger.info("python-docx unavailable for DOCX extraction: %s", exc)
        return []

    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        logger.info("DOCX section extraction failed: %s", exc)
        return []

    sections: List[Dict[str, Any]] = []
    current_text: List[str] = []
    current_images: List[str] = []
    total_images = 0

    def flush() -> None:
        text = "\n".join(current_text).strip()
        if text or current_images:
            sections.append({"text": text, "images": list(current_images)})
        current_text.clear()
        current_images.clear()

    try:
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                current_text.append(para.text.strip())
            if total_images >= max_images:
                continue
            for blip in para._p.findall(f".//{{{_DOCX_DRAWING_NS}}}blip"):
                rid = blip.get(f"{{{_DOCX_RELS_NS}}}embed")
                if not rid:
                    continue
                try:
                    part = doc.part.related_parts[rid]
                    img_bytes = part.blob
                    ext = Path(part.partname).suffix or ".png"
                except Exception:
                    continue
                if len(img_bytes) < 1024:  # skip tiny decorative images
                    continue
                saved = _save_media(img_bytes, ext)
                if saved and saved not in current_images:
                    current_images.append(saved)
                    total_images += 1
                    flush()
                if total_images >= max_images:
                    break
    except Exception as exc:
        logger.info("DOCX section extraction failed: %s", exc)
        return []
    flush()

    table_parts = []
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                table_parts.append(" | ".join(cells))
    if table_parts:
        sections.append({"text": "\n".join(table_parts), "images": []})

    return sections


def _extract_pptx(data: bytes) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise IngestError("بسته python-pptx نصب نیست.") from exc
    prs = Presentation(io.BytesIO(data))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_bits: list[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                t = shape.text.strip()
                if t:
                    slide_bits.append(t)
        if slide_bits:
            parts.append(f"اسلاید {i}:\n" + "\n".join(slide_bits))
    return "\n\n".join(parts).strip()


def _extract_xlsx(data: bytes) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise IngestError("بسته openpyxl نصب نیست.") from exc
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet in wb.worksheets:
        rows_out: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                rows_out.append(" | ".join(cells))
        if rows_out:
            parts.append(f"برگه {sheet.title}:\n" + "\n".join(rows_out[:500]))
    return "\n\n".join(parts).strip()


def _extract_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows: list[str] = []
    for i, row in enumerate(reader):
        if i >= 2000:
            rows.append("… (ادامه ردیف‌ها خلاصه شد)")
            break
        cells = [c.strip() for c in row if c and c.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows).strip()


def _extract_json(data: bytes) -> str:
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        raise IngestError("فایل JSON نامعتبر است.") from exc

    chunks: list[str] = []

    def walk(node, path: str = "") -> None:
        if isinstance(node, dict):
            # Treat each top-level / nested object as a record when small
            if path.count(".") <= 2 and all(
                not isinstance(v, (dict, list)) for v in node.values()
            ):
                bits = [f"{k}: {v}" for k, v in node.items() if v is not None]
                if bits:
                    chunks.append((" / ".join(path.split(".")) + ": " if path else "") + " | ".join(bits))
            else:
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, item in enumerate(node[:500]):
                walk(item, f"{path}[{i}]" if path else f"[{i}]")
        else:
            if node is not None and str(node).strip():
                chunks.append(f"{path}: {node}" if path else str(node))

    walk(obj)
    return "\n".join(chunks).strip()


async def _extract_audio(filename: str, data: bytes) -> str:
    """Transcribe audio with faster-whisper when available."""
    try:
        from ollama_rag.voice import transcribe_audio_bytes

        text = await transcribe_audio_bytes(data, filename=filename)
        if text and len(text.strip()) >= 8:
            return f"رونویسی صوت «{filename}»:\n{text.strip()}"
    except Exception as exc:
        logger.info("Audio transcription failed: %s", exc)
    return (
        f"فایل صوتی بارگذاری‌شده: {filename} ({len(data)} بایت). "
        "برای رونویسی، بسته faster-whisper را نصب و JAVAD_VOICE_ENABLED را فعال کنید."
    )

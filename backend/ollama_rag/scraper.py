"""
خزشگر وب محدود به دامنه — فقط لینک‌های داخلی همان دامنه را دنبال می‌کند.
متن هر صفحه پاکسازی می‌شود تا فقط محتوای اصلی وارد دانش جواد شود.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

ProgressCallback = Callable[[Dict], None]

import httpx
from bs4 import BeautifulSoup

from ollama_rag.text_clean import sanitize_document

# محدودیت‌های ایمنی برای جلوگیری از خزش بی‌پایان
DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_PAGES = 20
DEFAULT_TIMEOUT = 20.0
MAX_CHARS_PER_PAGE = 8000
MAX_TOTAL_CHARS = 60000

_SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "ftp:")
_SKIP_EXT = (
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".gz",
    ".tar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
)

_CHROME_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "svg",
    "iframe",
    "button",
    "template",
)


def _normalize_host(host: str) -> str:
    host = (host or "").lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _same_domain(url: str, root_host: str) -> bool:
    try:
        host = _normalize_host(urlparse(url).hostname or "")
    except Exception:
        return False
    if not host or not root_host:
        return False
    return host == root_host or host.endswith("." + root_host)


def _canonicalize(url: str) -> str:
    """یکسان‌سازی آدرس برای جلوگیری از بازدید تکراری."""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    # حذف پارامترهای رهگیری رایج
    query = parsed.query
    if query:
        keep = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].lower()
            if key in {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}:
                continue
            if part:
                keep.append(part)
        query = "&".join(keep)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _looks_like_asset(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(path.endswith(ext) for ext in _SKIP_EXT)


def _extract_internal_links(soup: BeautifulSoup, base_url: str, root_host: str) -> List[str]:
    found: List[str] = []
    seen_local: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#"):
            continue
        low = href.lower()
        if any(low.startswith(s) for s in _SKIP_SCHEMES):
            continue
        absolute = urljoin(base_url, href)
        absolute = _canonicalize(absolute)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not _same_domain(absolute, root_host):
            continue
        if _looks_like_asset(absolute):
            continue
        if absolute in seen_local:
            continue
        seen_local.add(absolute)
        found.append(absolute)
    return found


def _strip_chrome(soup: BeautifulSoup) -> None:
    """حذف ناوبری و قالب تکراری — ابتدا جمع‌آوری، سپس حذف (بدون تغییر همزمان درخت)."""
    to_drop = []

    for tag in soup(list(_CHROME_TAGS)):
        to_drop.append(tag)

    for a in soup.find_all("a"):
        label = (a.get_text() or "").strip()
        low = label.lower()
        if "پرش" in label or "skip" in low or "رفتن به" in label:
            to_drop.append(a)

    junk_re = re.compile(
        r"(nav|menu|footer|header|sidebar|breadcrumb|cookie|banner|social|share|toolbar)",
        re.I,
    )
    for el in soup.find_all(True):
        # پس از تجزیهٔ والد، attrs ممکن است None شود
        if el is None or getattr(el, "attrs", None) is None:
            continue
        parts = []
        for k in ("id", "class", "role"):
            val = el.attrs.get(k)
            if not val:
                continue
            if isinstance(val, (list, tuple)):
                parts.append(" ".join(str(x) for x in val))
            else:
                parts.append(str(val))
        attrs = " ".join(parts)
        if attrs and junk_re.search(attrs):
            to_drop.append(el)

    seen = set()
    for el in to_drop:
        if el is None or id(el) in seen:
            continue
        seen.add(id(el))
        try:
            el.decompose()
        except Exception:
            continue


def _page_text(soup: BeautifulSoup) -> Tuple[str, str]:
    title_raw = ""
    if soup.title and soup.title.string:
        title_raw = soup.title.string
    title = sanitize_document(title_raw) or "صفحه وب"

    work = BeautifulSoup(str(soup), "html.parser")
    _strip_chrome(work)

    root = (
        work.find("main")
        or work.find("article")
        or work.find(attrs={"role": "main"})
        or work.find(class_=re.compile(r"(content|post|entry|article)", re.I))
    )
    if root is None:
        root = work.body or work

    raw = root.get_text(" ", strip=True) if root else ""
    text = sanitize_document(raw)
    words = text.split()
    if len(words) > 20:
        while words and len(words[0]) <= 2:
            words.pop(0)
        text = " ".join(words)
    if len(text) > MAX_CHARS_PER_PAGE:
        text = text[:MAX_CHARS_PER_PAGE].rsplit(" ", 1)[0]
    return title, text


def _fetch_html(client: httpx.Client, url: str) -> str:
    try:
        r = client.get(url)
        r.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ValueError("زمان دریافت صفحه به پایان رسید.") from exc
    except httpx.HTTPStatusError as exc:
        raise ValueError(f"پاسخ ناموفق از وب‌سایت (کد {exc.response.status_code}).") from exc
    except httpx.RequestError as exc:
        raise ValueError("امکان اتصال به وب‌سایت وجود ندارد.") from exc

    ctype = (r.headers.get("content-type") or "").lower()
    if "html" not in ctype and "text/" not in ctype and ctype:
        # بعضی سرورها نوع محتوا نمی‌فرستند؛ فقط انواع غیرمتنی را رد کن
        if any(x in ctype for x in ("image/", "audio/", "video/", "application/pdf", "application/zip")):
            raise ValueError("این آدرس یک صفحهٔ متنی نیست.")
    return r.content.decode(r.encoding or "utf-8", errors="ignore")


def _emit(on_progress: Optional[ProgressCallback], payload: Dict) -> None:
    if not on_progress:
        return
    try:
        on_progress(payload)
    except Exception:
        pass


def scrape_website(
    url: str,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = DEFAULT_TIMEOUT,
    on_progress: Optional[ProgressCallback] = None,
) -> dict:
    """
    خزش بازگشتی محدود به دامنهٔ ورودی.
    فقط لینک‌های <a> که به همان دامنه می‌روند دنبال می‌شوند.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("لطفاً آدرس وب‌سایت را وارد کنید.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("آدرس وب‌سایت نامعتبر است.")

    start = _canonicalize(url)
    root_host = _normalize_host(parsed.hostname or "")
    if not root_host:
        raise ValueError("دامنهٔ وب‌سایت نامعتبر است.")

    max_depth = max(0, min(int(max_depth), 4))
    max_pages = max(1, min(int(max_pages), 40))

    visited: Set[str] = set()
    queue: deque[Tuple[str, int]] = deque([(start, 0)])
    pages: List[Dict[str, str]] = []
    sections: List[str] = []
    crawl_log: List[Dict[str, object]] = []
    site_title = root_host
    attempts = 0

    def log_link(
        *,
        link: str,
        status: str,
        depth: int = 0,
        title: str = "",
        message: str = "",
        percent: Optional[int] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
    ) -> None:
        entry = {
            "url": link,
            "status": status,
            "depth": depth,
            "title": title or "",
            "domain": root_host,
        }
        crawl_log.append(entry)
        payload: Dict = {
            "phase": "website",
            "event": "link",
            "current": current if current is not None else len(pages),
            "total": total if total is not None else max_pages,
            "percent": percent if percent is not None else min(95, int(100 * len(pages) / max(max_pages, 1))),
            "message": message or f"پیمایش: {link}",
            "link": entry,
            "links_count": len(crawl_log),
        }
        _emit(on_progress, payload)

    _emit(
        on_progress,
        {
            "phase": "website",
            "current": 0,
            "total": max_pages,
            "percent": 2,
            "message": f"شروع پیمایش دامنهٔ «{root_host}»…",
            "domain": root_host,
        },
    )

    headers = {
        "User-Agent": "JavadLocalCrawler/2.0 (+local; fa)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fa,en;q=0.5",
    }

    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        while queue and len(pages) < max_pages:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            attempts += 1

            remaining = len(queue) + 1
            est_total = min(max_pages, max(len(pages) + remaining, attempts))
            pct = min(95, int(100 * max(len(pages), attempts - 1) / max(est_total, 1)))
            log_link(
                link=current,
                status="visiting",
                depth=depth,
                message=f"در حال خواندن ({attempts}): {current}",
                percent=max(5, pct),
                current=len(pages),
                total=est_total,
            )

            try:
                html = _fetch_html(client, current)
            except ValueError:
                log_link(
                    link=current,
                    status="failed",
                    depth=depth,
                    message=f"خواندن ناموفق: {current}",
                    percent=max(5, pct),
                    current=len(pages),
                    total=est_total,
                )
                if not pages and current == start:
                    raise
                continue

            soup = BeautifulSoup(html, "html.parser")
            title, text = _page_text(soup)

            if len(text) >= 40:
                pages.append({"url": current, "title": title, "text": text})
                if not sections:
                    site_title = title or root_host
                block = f"بخش: {title}\n{text}" if title else text
                sections.append(block)
                remaining = len(queue)
                est_total = min(max_pages, max(len(pages) + remaining, len(pages)))
                pct = min(98, int(100 * len(pages) / max(est_total, 1)))
                short = (title[:42] + "…") if len(title) > 42 else title
                log_link(
                    link=current,
                    status="learned",
                    depth=depth,
                    title=title,
                    message=f"آموخته شد: {short}",
                    percent=pct,
                    current=len(pages),
                    total=est_total,
                )
            else:
                log_link(
                    link=current,
                    status="skipped",
                    depth=depth,
                    title=title,
                    message=f"متن کافی نبود؛ رد شد: {current}",
                    percent=max(5, pct),
                    current=len(pages),
                    total=est_total,
                )

            if depth < max_depth and len(pages) < max_pages:
                for link in _extract_internal_links(soup, current, root_host):
                    if link not in visited:
                        queue.append((link, depth + 1))

    if not sections:
        raise ValueError("متن قابل‌توجهی از این وب‌سایت استخراج نشد.")

    _emit(
        on_progress,
        {
            "phase": "website",
            "current": len(pages),
            "total": len(pages),
            "percent": 99,
            "message": "در حال ذخیره در دانش جواد…",
            "domain": root_host,
            "crawl_log": crawl_log,
        },
    )

    combined = "\n\n".join(sections)
    combined = sanitize_document(combined)
    if len(combined) > MAX_TOTAL_CHARS:
        combined = combined[:MAX_TOTAL_CHARS].rsplit(" ", 1)[0]

    return {
        "url": start,
        "title": site_title or "وب‌سایت",
        "text": combined,
        "chars": len(combined),
        "pages": len(pages),
        "domain": root_host,
        "page_titles": [p["title"] for p in pages],
        "crawl_log": crawl_log,
        "learned_urls": [p["url"] for p in pages],
    }


# سازگاری با فراخوانی‌های قدیمی
def scrape_url(url: str, **kwargs) -> dict:
    return scrape_website(url, **kwargs)

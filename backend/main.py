"""
دستیار عامل جواد — سامانهٔ محلی با اولاما
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ollama_rag import client as ollama_client
from ollama_rag.agent import get_agent
from ollama_rag.auth import require_user
from ollama_rag.client import OllamaError
from ollama_rag.config import (
    CORS_ORIGINS,
    DEFAULT_LANGUAGE,
    EMBED_MODEL,
    CHAT_MODEL,
)
from ollama_rag.logging_setup import RequestIdMiddleware, setup_logging
from ollama_rag.metrics import metrics
from ollama_rag.scraper import (
    mark_recrawl_done,
    register_recrawl_target,
    scrape_website,
)
from ollama_rag.store import get_store
from ollama_rag.text_clean import sanitize_document
from ollama_rag.voice import (
    EnergyVAD,
    pcm16_to_wav,
    synthesize_speech,
    transcribe_audio_bytes,
    voice_status,
)
from rag.ingest import IngestError, ingest_file

setup_logging()
logger = logging.getLogger("javad.main")

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "دستیار عامل جواد"

app = FastAPI(
    title=APP_TITLE,
    description="گفتگوی فارسی محلی با اولاما و یادگیری اسناد و وب‌سایت (خزش محدود به دامنه)",
    version="5.0.0",
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- simple in-process rate limiter ---
_rate_buckets: Dict[str, List[float]] = {}
_rate_lock = asyncio.Lock()


async def _rate_limit(key: str, limit: int, window_s: float = 60.0) -> None:
    import time

    now = time.time()
    async with _rate_lock:
        bucket = _rate_buckets.setdefault(key, [])
        _rate_buckets[key] = [t for t in bucket if now - t < window_s]
        if len(_rate_buckets[key]) >= limit:
            raise HTTPException(status_code=429, detail="تعداد درخواست‌ها بیش از حد مجاز است. کمی صبر کنید.")
        _rate_buckets[key].append(now)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="پیام کاربر")
    session_id: str = Field(default="default")
    language: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str = Field(..., description="پاسخ طبیعی جواد")
    context_used: bool = False
    chunks_used: int = 0
    mode: str = "ollama"
    sources: List[str] = Field(default_factory=list)
    citations: List[dict] = Field(default_factory=list)
    language: str = DEFAULT_LANGUAGE


class WebsiteRequest(BaseModel):
    url: str = Field(..., min_length=8)
    max_depth: int = Field(default=2, ge=0, le=6)
    max_pages: int = Field(default=20, ge=1, le=100)
    use_sitemap: bool = True
    respect_robots: bool = True
    schedule_recrawl_hours: Optional[float] = Field(default=None, ge=1)


class IngestResult(BaseModel):
    source: str
    kind: str
    chars: int
    chunks_added: int
    message: str
    pages: int = 0
    domain: str = ""
    crawl_log: List[dict] = Field(default_factory=list)
    learned_urls: List[str] = Field(default_factory=list)


def _website_ingest_message(scraped: dict) -> str:
    if scraped.get("unchanged"):
        return "محتوای وب‌سایت نسبت به قبل تغییری نداشته؛ دانش به‌روز است."
    pages = int(scraped.get("pages") or 1)
    title = scraped.get("title") or "وب‌سایت"
    domain = scraped.get("domain") or ""
    if pages > 1:
        domain_bit = f" در دامنهٔ «{domain}»" if domain else ""
        return (
            f"جواد از «{title}» یاد گرفت "
            f"({pages} صفحهٔ داخلی{domain_bit}؛ فقط همان دامنه)."
        )
    return f"جواد از وب‌سایت «{title}» یاد گرفت."


def _ndjson(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _prev_hashes(user_id: str) -> Dict[str, str]:
    store = get_store()
    # Reconstruct from crawl_pages table via raw query helper
    out: Dict[str, str] = {}
    try:
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT url, content_hash FROM crawl_pages WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            for r in rows:
                out[r["url"]] = r["content_hash"]
    except Exception:
        pass
    return out


def _save_page_hashes(scraped: dict, user_id: str) -> None:
    store = get_store()
    for url, h in (scraped.get("page_hashes") or {}).items():
        store.set_crawl_page(url, h, title=scraped.get("title") or "", user_id=user_id)


async def _ingest_files_with_progress(
    files: List[UploadFile],
    user_id: str,
) -> AsyncIterator[str]:
    agent = get_agent(user_id)
    valid = [f for f in files if f.filename]
    total = len(valid)
    results: List[dict] = []
    if total == 0:
        return

    yield _ndjson(
        {
            "type": "progress",
            "phase": "file",
            "current": 0,
            "total": total,
            "percent": 2,
            "message": f"شروع یادگیری {total} فایل…",
        }
    )

    for idx, upload in enumerate(valid, start=1):
        name = upload.filename or "فایل"
        yield _ndjson(
            {
                "type": "progress",
                "phase": "file",
                "current": idx - 1,
                "total": total,
                "percent": int(100 * (idx - 1) / total) if total else 0,
                "message": f"در حال خواندن «{name}»…",
            }
        )
        data = await upload.read()
        if not data:
            continue
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{Path(name).name}"
        dest.write_bytes(data)
        try:
            extracted = await ingest_file(name, data)
        except IngestError as exc:
            yield _ndjson({"type": "error", "message": str(exc)})
            return
        except Exception:
            yield _ndjson({"type": "error", "message": f"خطا در خواندن فایل «{name}»."})
            return
        cleaned = sanitize_document(extracted.get("text") or "")
        added = agent.ingest(cleaned, extracted["source"], extracted["kind"])
        item = IngestResult(
            source=extracted["source"],
            kind=extracted["kind"],
            chars=len(cleaned),
            chunks_added=added,
            message=(
                f"«{name}» به دانش جواد اضافه شد."
                if added
                else f"«{name}» تکراری بود و اضافه نشد."
            ),
            pages=0,
        )
        results.append(item.model_dump())
        progress_label = (
            f"آموخته شد: «{name}» ({idx} از {total})"
            if added
            else f"تکراری بود، رد شد: «{name}» ({idx} از {total})"
        )
        yield _ndjson(
            {
                "type": "progress",
                "phase": "file",
                "current": idx,
                "total": total,
                "percent": int(100 * idx / total),
                "message": progress_label,
            }
        )
        yield _ndjson({"type": "item", "result": item.model_dump()})

    if results:
        yield _ndjson(
            {
                "type": "done",
                "results": results,
                "message": " · ".join(r["message"] for r in results),
            }
        )


async def _ingest_website_with_progress(
    url: str,
    user_id: str,
    *,
    max_depth: int = 2,
    max_pages: int = 20,
    use_sitemap: bool = True,
    respect_robots: bool = True,
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    prev = _prev_hashes(user_id)

    def on_progress(info: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", info))

    def run_scrape() -> dict:
        return scrape_website(
            url,
            max_depth=max_depth,
            max_pages=max_pages,
            use_sitemap=use_sitemap,
            respect_robots=respect_robots,
            previous_hashes=prev,
            on_progress=on_progress,
            user_id=user_id,
        )

    task = asyncio.create_task(asyncio.to_thread(run_scrape))

    while True:
        if task.done() and queue.empty():
            break
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=0.15)
        except asyncio.TimeoutError:
            if task.done():
                break
            continue
        if kind == "progress":
            yield _ndjson({"type": "progress", **payload})

    try:
        scraped = await task
    except ValueError as exc:
        yield _ndjson({"type": "error", "message": str(exc)})
        return
    except Exception:
        yield _ndjson({"type": "error", "message": "خطا در پیمایش وب‌سایت."})
        return

    while not queue.empty():
        kind, payload = queue.get_nowait()
        if kind == "progress":
            yield _ndjson({"type": "progress", **payload})

    _save_page_hashes(scraped, user_id)

    yield _ndjson(
        {
            "type": "progress",
            "phase": "website",
            "current": scraped.get("pages") or 1,
            "total": scraped.get("pages") or 1,
            "percent": 99,
            "message": "در حال ذخیره در دانش جواد…",
            "crawl_log": scraped.get("crawl_log") or [],
            "domain": scraped.get("domain") or "",
        }
    )

    added = 0
    if scraped.get("text"):
        added = get_agent(user_id).ingest(scraped["text"], scraped["title"], "وبسایت")

    item = IngestResult(
        source=scraped["title"],
        kind="وبسایت",
        chars=scraped["chars"],
        chunks_added=added,
        message=_website_ingest_message(scraped),
        pages=int(scraped.get("pages") or 0),
        domain=str(scraped.get("domain") or ""),
        crawl_log=list(scraped.get("crawl_log") or []),
        learned_urls=list(scraped.get("learned_urls") or []),
    )
    yield _ndjson(
        {
            "type": "progress",
            "phase": "website",
            "current": item.pages,
            "total": item.pages or 1,
            "percent": 100,
            "message": item.message,
            "crawl_log": item.crawl_log,
            "domain": item.domain,
        }
    )
    yield _ndjson({"type": "item", "result": item.model_dump()})
    yield _ndjson({"type": "done", "results": [item.model_dump()], "message": item.message})


@app.post("/api/rag/upload", response_model=List[IngestResult])
async def rag_upload(
    request: Request,
    files: Optional[List[UploadFile]] = File(default=None),
    url: Optional[str] = Form(default=None),
    max_depth: int = Form(default=2),
    max_pages: int = Form(default=20),
    user_id: str = Depends(require_user),
):
    """بارگذاری فایل یا آدرس وب؛ وب‌سایت با خزش بازگشتی محدود به دامنه آموخته می‌شود."""
    await _rate_limit(f"upload:{user_id}", 10)
    agent = get_agent(user_id)
    results: List[IngestResult] = []

    if not files and not (url and url.strip()):
        raise HTTPException(status_code=400, detail="حداقل یک فایل یا یک آدرس وب‌سایت بفرستید.")

    if files:
        for upload in files:
            if not upload.filename:
                continue
            data = await upload.read()
            if not data:
                continue
            dest = UPLOADS_DIR / f"{uuid.uuid4().hex}_{Path(upload.filename).name}"
            dest.write_bytes(data)
            try:
                extracted = await ingest_file(upload.filename, data)
            except IngestError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"خطا در خواندن فایل «{upload.filename}».",
                ) from exc
            cleaned = sanitize_document(extracted.get("text") or "")
            added = agent.ingest(cleaned, extracted["source"], extracted["kind"])
            results.append(
                IngestResult(
                    source=extracted["source"],
                    kind=extracted["kind"],
                    chars=len(cleaned),
                    chunks_added=added,
                    message=(
                        f"«{upload.filename}» به دانش جواد اضافه شد."
                        if added
                        else f"«{upload.filename}» تکراری بود و اضافه نشد."
                    ),
                    pages=0,
                )
            )

    if url and url.strip():
        await _rate_limit(f"scrape:{user_id}", 5)
        try:
            scraped = scrape_website(
                url.strip(),
                max_depth=max_depth,
                max_pages=max_pages,
                previous_hashes=_prev_hashes(user_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _save_page_hashes(scraped, user_id)
        added = 0
        if scraped.get("text"):
            added = agent.ingest(scraped["text"], scraped["title"], "وبسایت")
        results.append(
            IngestResult(
                source=scraped["title"],
                kind="وبسایت",
                chars=scraped["chars"],
                chunks_added=added,
                message=_website_ingest_message(scraped),
                pages=int(scraped.get("pages") or 0),
                domain=str(scraped.get("domain") or ""),
                crawl_log=list(scraped.get("crawl_log") or []),
                learned_urls=list(scraped.get("learned_urls") or []),
            )
        )

    if not results:
        raise HTTPException(status_code=400, detail="محتوای معتبری دریافت نشد.")
    return results


@app.post("/api/rag/upload/stream")
async def rag_upload_stream(
    files: Optional[List[UploadFile]] = File(default=None),
    url: Optional[str] = Form(default=None),
    max_depth: int = Form(default=2),
    max_pages: int = Form(default=20),
    user_id: str = Depends(require_user),
):
    """جریان پیشرفت یادگیری (فایل و/یا وب‌سایت) به صورت خطوط جیسون."""
    await _rate_limit(f"upload:{user_id}", 10)
    file_list = [f for f in (files or []) if f and f.filename]
    has_url = bool(url and url.strip())
    if not file_list and not has_url:
        raise HTTPException(status_code=400, detail="حداقل یک فایل یا یک آدرس وب‌سایت بفرستید.")

    async def event_stream() -> AsyncIterator[str]:
        collected: List[dict] = []
        dual = bool(file_list and has_url)

        def remap(evt: dict) -> dict:
            if not dual or evt.get("type") != "progress":
                return evt
            pct = float(evt.get("percent") or 0)
            phase = evt.get("phase")
            if phase == "file":
                evt = {**evt, "percent": min(48, int(pct * 0.48))}
            elif phase == "website":
                evt = {**evt, "percent": min(99, 48 + int(pct * 0.5))}
            return evt

        if file_list:
            async for line in _ingest_files_with_progress(file_list, user_id):
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue
                kind = evt.get("type")
                if kind == "error":
                    yield line
                    return
                if kind == "done":
                    collected.extend(evt.get("results") or [])
                    continue
                yield _ndjson(remap(evt))

        if has_url:
            await _rate_limit(f"scrape:{user_id}", 5)
            async for line in _ingest_website_with_progress(
                url.strip(),
                user_id,
                max_depth=max_depth,
                max_pages=max_pages,
            ):
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue
                kind = evt.get("type")
                if kind == "error":
                    yield line
                    return
                if kind == "done":
                    collected.extend(evt.get("results") or [])
                    continue
                yield _ndjson(remap(evt))

        if not collected:
            yield _ndjson({"type": "error", "message": "محتوای معتبری دریافت نشد."})
            return

        yield _ndjson(
            {
                "type": "progress",
                "phase": "done",
                "current": 1,
                "total": 1,
                "percent": 100,
                "message": "یادگیری کامل شد.",
            }
        )
        yield _ndjson(
            {
                "type": "done",
                "results": collected,
                "message": " · ".join(
                    r.get("message", "") for r in collected if r.get("message")
                ),
            }
        )

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/website/scrape", response_model=IngestResult)
async def website_scrape(body: WebsiteRequest, user_id: str = Depends(require_user)):
    """یادگیری از وب‌سایت با خزش بازگشتی محدود به دامنه و پاکسازی متن."""
    await _rate_limit(f"scrape:{user_id}", 5)
    try:
        scraped = scrape_website(
            body.url,
            max_depth=body.max_depth,
            max_pages=body.max_pages,
            use_sitemap=body.use_sitemap,
            respect_robots=body.respect_robots,
            previous_hashes=_prev_hashes(user_id),
            user_id=user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _save_page_hashes(scraped, user_id)
    if body.schedule_recrawl_hours:
        register_recrawl_target(
            body.url,
            max_depth=body.max_depth,
            max_pages=body.max_pages,
            user_id=user_id,
            interval_hours=body.schedule_recrawl_hours,
        )
    added = 0
    if scraped.get("text"):
        added = get_agent(user_id).ingest(scraped["text"], scraped["title"], "وبسایت")
    return IngestResult(
        source=scraped["title"],
        kind="وبسایت",
        chars=scraped["chars"],
        chunks_added=added,
        message=_website_ingest_message(scraped),
        pages=int(scraped.get("pages") or 0),
        domain=str(scraped.get("domain") or ""),
        crawl_log=list(scraped.get("crawl_log") or []),
        learned_urls=list(scraped.get("learned_urls") or []),
    )


@app.post("/api/ollama/chat", response_model=ChatResponse)
async def ollama_chat(body: ChatRequest, user_id: str = Depends(require_user)):
    """گفتگو با اولاما + زمینه دانش."""
    await _rate_limit(f"chat:{user_id}", 60)
    result = await get_agent(user_id).chat(
        body.prompt,
        session_id=body.session_id or "default",
        language=body.language,
        model=body.model,
    )
    answer = (result.get("answer") or "").strip()
    if not answer:
        answer = "الان نتوانستم پاسخ روشنی بدهم. لطفاً سؤال را کمی واضح‌تر بپرسید."
    return ChatResponse(
        answer=answer,
        context_used=bool(result.get("context_used")),
        chunks_used=int(result.get("chunks_used") or 0),
        mode=result.get("mode") or "ollama",
        sources=list(result.get("sources") or []),
        citations=list(result.get("citations") or []),
        language=result.get("language") or DEFAULT_LANGUAGE,
    )


@app.post("/api/ollama/chat/stream")
async def ollama_chat_stream(body: ChatRequest, user_id: str = Depends(require_user)):
    """جریان توکن‌به‌توکن پاسخ گفتگو."""
    await _rate_limit(f"chat:{user_id}", 60)

    async def event_stream() -> AsyncIterator[str]:
        async for evt in get_agent(user_id).chat_stream(
            body.prompt,
            session_id=body.session_id or "default",
            language=body.language,
            model=body.model,
        ):
            yield _ndjson(evt)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/memory/clear")
@app.delete("/api/memory/clear")
async def memory_clear(user_id: str = Depends(require_user)):
    """پاکسازی کامل دانش و حافظهٔ گفتگو برای کاربر جاری."""
    get_agent(user_id).clear()
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "message": "حافظهٔ جواد پاک شد. تمام اسناد، وب‌سایت‌ها و گفتگوهای قبلی حذف شدند.",
    }


@app.get("/api/knowledge")
async def list_knowledge(user_id: str = Depends(require_user)):
    """فهرست منابع دانش کاربر."""
    sources = get_agent(user_id).list_sources()
    return {"sources": sources, "count": len(sources)}


@app.delete("/api/rag/source/{source_id}")
@app.delete("/api/knowledge/{source_id}")
async def delete_source(source_id: str, user_id: str = Depends(require_user)):
    result = get_agent(user_id).delete_source(source_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="منبع پیدا نشد.")
    return {
        "ok": True,
        "message": f"منبع «{result.get('source')}» حذف شد.",
        "deleted_chunks": result.get("deleted_chunks", 0),
    }


@app.post("/api/reindex")
async def reindex(user_id: str = Depends(require_user)):
    """بازسازی بردارهای امبدینگ برای تمام قطعه‌های دانش."""
    result = await get_agent(user_id).reindex()
    return {"ok": True, **result}


@app.websocket("/ws/voip-stream")
async def voip_stream(websocket: WebSocket):
    """جریان صوت به متن (faster-whisper) + پاسخ جواد + TTS اختیاری (Piper)."""
    await websocket.accept()
    # Auth via query param when API key is configured
    from ollama_rag.config import API_KEY

    api_key = websocket.query_params.get("api_key") or ""
    user_id = websocket.query_params.get("user_id") or "default"
    language = websocket.query_params.get("language") or DEFAULT_LANGUAGE
    session_id = websocket.query_params.get("session_id") or "voice"

    if API_KEY and api_key != API_KEY:
        await websocket.send_json({"type": "error", "message": "کلید API نامعتبر است."})
        await websocket.close(code=4401)
        return

    status = voice_status()
    await websocket.send_json(
        {
            "type": "ready",
            "message": "کانال صوت آماده است.",
            "voice": status,
        }
    )

    vad = EnergyVAD()
    agent = get_agent(user_id)

    async def handle_utterance(pcm: bytes) -> None:
        wav = pcm16_to_wav(pcm)
        await websocket.send_json({"type": "stt_start"})
        text = await transcribe_audio_bytes(wav, filename="utterance.wav", language=language)
        if not text:
            await websocket.send_json(
                {"type": "info", "message": "گفتاری تشخیص داده نشد یا موتور STT در دسترس نیست."}
            )
            return
        await websocket.send_json({"type": "transcript", "text": text})
        result = await agent.chat(text, session_id=session_id, language=language)
        answer = result.get("answer") or ""
        await websocket.send_json(
            {
                "type": "answer",
                "text": answer,
                "sources": result.get("sources") or [],
                "citations": result.get("citations") or [],
                "mode": result.get("mode"),
            }
        )
        audio = await synthesize_speech(answer, language=language)
        if audio:
            await websocket.send_bytes(audio)
            await websocket.send_json({"type": "tts_done"})

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                utterances = vad.feed(message["bytes"])
                for utt in utterances:
                    await handle_utterance(utt)
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "پیام نامعتبر است."})
                continue
            t = data.get("type", "")
            if t == "ping":
                await websocket.send_json({"type": "pong"})
            elif t == "flush":
                leftover = vad.flush()
                if leftover:
                    await handle_utterance(leftover)
            elif t == "text":
                # Allow text fallback over the same channel
                prompt = (data.get("text") or "").strip()
                if prompt:
                    result = await agent.chat(prompt, session_id=session_id, language=language)
                    await websocket.send_json(
                        {
                            "type": "answer",
                            "text": result.get("answer") or "",
                            "sources": result.get("sources") or [],
                            "citations": result.get("citations") or [],
                        }
                    )
            elif t == "close":
                leftover = vad.flush()
                if leftover:
                    await handle_utterance(leftover)
                await websocket.close()
                break
            else:
                await websocket.send_json(
                    {"type": "info", "message": "نوع پیام پشتیبانی نمی‌شود."}
                )
    except WebSocketDisconnect:
        return


@app.get("/api/health")
async def health():
    agent = get_agent("default")
    ollama_ok = False
    models: List[str] = []
    try:
        info = await ollama_client.check_ollama()
        ollama_ok = True
        models = info.get("models") or []
    except OllamaError:
        pass
    embed_info = await ollama_client.check_embed_model()
    st = agent.status()
    return {
        "status": "ok",
        "name": APP_TITLE,
        "chunks": st["chunks"],
        "sources": st["sources"],
        "ollama": ollama_ok,
        "models": models,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "embed_ok": embed_info.get("ok"),
        "vector_store": st["vector_backend"],
        "vector_count": st["vector_count"],
        "voice": voice_status(),
        "metrics": metrics.snapshot(),
        "voip": "/ws/voip-stream",
        "default_language": DEFAULT_LANGUAGE,
    }


@app.get("/api/metrics")
async def get_metrics(user_id: str = Depends(require_user)):
    return metrics.snapshot()


@app.get("/api/config")
async def public_config():
    """Public UI config (no secrets)."""
    return {
        "default_language": DEFAULT_LANGUAGE,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "languages": ["fa", "en", "ar"],
        "auth_required": bool(__import__("ollama_rag.config", fromlist=["API_KEY"]).API_KEY)
        or __import__("ollama_rag.config", fromlist=["AUTH_REQUIRED"]).AUTH_REQUIRED,
    }


@app.get("/")
async def index():
    path = FRONTEND_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="رابط کاربری پیدا نشد.")
    return FileResponse(path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.on_event("startup")
async def _startup() -> None:
    # Eager-load default agent so SQLite/Chroma are warm
    get_agent("default")
    logger.info("JAVAD started — model=%s embed=%s", CHAT_MODEL, EMBED_MODEL)

    async def _recrawl_loop() -> None:
        from ollama_rag.scraper import due_recrawl_targets

        while True:
            await asyncio.sleep(3600)
            for target in due_recrawl_targets():
                try:
                    uid = target.get("user_id") or "default"
                    scraped = await asyncio.to_thread(
                        scrape_website,
                        target["url"],
                        max_depth=target.get("max_depth", 2),
                        max_pages=target.get("max_pages", 20),
                        previous_hashes=_prev_hashes(uid),
                        user_id=uid,
                    )
                    _save_page_hashes(scraped, uid)
                    if scraped.get("text"):
                        get_agent(uid).ingest(scraped["text"], scraped["title"], "وبسایت")
                    mark_recrawl_done(target["url"], uid)
                    logger.info("Re-crawled %s for user %s", target["url"], uid)
                except Exception:
                    logger.exception("Re-crawl failed for %s", target.get("url"))

    asyncio.create_task(_recrawl_loop())

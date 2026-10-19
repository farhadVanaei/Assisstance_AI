"""
دستیار عامل جواد — سامانهٔ محلی با اولاما
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ollama_rag import client as ollama_client
from ollama_rag.agent import get_agent
from ollama_rag.client import OllamaError
from ollama_rag.scraper import scrape_website
from ollama_rag.text_clean import sanitize_document
from rag.ingest import ingest_file

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "دستیار عامل جواد"

app = FastAPI(
    title=APP_TITLE,
    description="گفتگوی فارسی محلی با اولاما و یادگیری اسناد و وب‌سایت (خزش محدود به دامنه)",
    version="4.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="پیام فارسی کاربر")
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="پاسخ طبیعی جواد به فارسی")
    context_used: bool = False
    chunks_used: int = 0
    mode: str = "ollama"


class WebsiteRequest(BaseModel):
    url: str = Field(..., min_length=8)


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


async def _ingest_files_with_progress(
    files: List[UploadFile],
) -> AsyncIterator[str]:
    agent = get_agent()
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
        except Exception:
            yield _ndjson(
                {
                    "type": "error",
                    "message": f"خطا در خواندن فایل «{name}».",
                }
            )
            return
        cleaned = sanitize_document(extracted.get("text") or "")
        added = agent.ingest(cleaned, extracted["source"], extracted["kind"])
        item = IngestResult(
            source=extracted["source"],
            kind=extracted["kind"],
            chars=len(cleaned),
            chunks_added=added,
            message=f"«{name}» به دانش جواد اضافه شد.",
            pages=0,
        )
        results.append(item.model_dump())
        yield _ndjson(
            {
                "type": "progress",
                "phase": "file",
                "current": idx,
                "total": total,
                "percent": int(100 * idx / total),
                "message": f"آموخته شد: «{name}» ({idx} از {total})",
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


async def _ingest_website_with_progress(url: str) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_progress(info: Dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", info))

    def run_scrape() -> dict:
        return scrape_website(url, on_progress=on_progress)

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

    # drain leftover progress events
    while not queue.empty():
        kind, payload = queue.get_nowait()
        if kind == "progress":
            yield _ndjson({"type": "progress", **payload})

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
    added = get_agent().ingest(scraped["text"], scraped["title"], "وبسایت")
    item = IngestResult(
        source=scraped["title"],
        kind="وبسایت",
        chars=scraped["chars"],
        chunks_added=added,
        message=_website_ingest_message(scraped),
        pages=int(scraped.get("pages") or 1),
        domain=str(scraped.get("domain") or ""),
        crawl_log=list(scraped.get("crawl_log") or []),
        learned_urls=list(scraped.get("learned_urls") or []),
    )
    yield _ndjson(
        {
            "type": "progress",
            "phase": "website",
            "current": item.pages,
            "total": item.pages,
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
    files: Optional[List[UploadFile]] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    """بارگذاری فایل یا آدرس وب؛ وب‌سایت با خزش بازگشتی محدود به دامنه آموخته می‌شود."""
    agent = get_agent()
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
                    message=f"«{upload.filename}» به دانش جواد اضافه شد.",
                    pages=0,
                )
            )

    if url and url.strip():
        try:
            scraped = scrape_website(url.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        added = agent.ingest(scraped["text"], scraped["title"], "وبسایت")
        results.append(
            IngestResult(
                source=scraped["title"],
                kind="وبسایت",
                chars=scraped["chars"],
                chunks_added=added,
                message=_website_ingest_message(scraped),
                pages=int(scraped.get("pages") or 1),
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
):
    """جریان پیشرفت یادگیری (فایل و/یا وب‌سایت) به صورت خطوط جیسون."""
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
            async for line in _ingest_files_with_progress(file_list):
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
            async for line in _ingest_website_with_progress(url.strip()):
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/website/scrape", response_model=IngestResult)
async def website_scrape(body: WebsiteRequest):
    """یادگیری از وب‌سایت با خزش بازگشتی محدود به دامنه و پاکسازی متن."""
    try:
        scraped = scrape_website(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    added = get_agent().ingest(scraped["text"], scraped["title"], "وبسایت")
    return IngestResult(
        source=scraped["title"],
        kind="وبسایت",
        chars=scraped["chars"],
        chunks_added=added,
        message=_website_ingest_message(scraped),
        pages=int(scraped.get("pages") or 1),
        domain=str(scraped.get("domain") or ""),
        crawl_log=list(scraped.get("crawl_log") or []),
        learned_urls=list(scraped.get("learned_urls") or []),
    )


@app.post("/api/ollama/chat", response_model=ChatResponse)
async def ollama_chat(body: ChatRequest):
    """
    گفتگوی فارسی با اولاما.
    همیشه یک رشتهٔ پاسخ طبیعی برمی‌گرداند (بدون قالب شکسته یا کد خام).
    """
    result = await get_agent().chat(body.prompt, session_id=body.session_id or "default")
    answer = (result.get("answer") or "").strip()
    if not answer:
        answer = "الان نتوانستم پاسخ روشنی بدهم. لطفاً سؤال را کمی واضح‌تر بپرسید."
    return ChatResponse(
        answer=answer,
        context_used=bool(result.get("context_used")),
        chunks_used=int(result.get("chunks_used") or 0),
        mode=result.get("mode") or "ollama",
    )


@app.post("/api/memory/clear")
@app.delete("/api/memory/clear")
async def memory_clear():
    """پاکسازی کامل دانش و حافظهٔ گفتگو."""
    get_agent().clear()
    if UPLOADS_DIR.exists():
        shutil.rmtree(UPLOADS_DIR)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "ok": True,
        "message": "حافظهٔ جواد پاک شد. تمام اسناد، وب‌سایت‌ها و گفتگوهای قبلی حذف شدند.",
    }


@app.websocket("/ws/voip-stream")
async def voip_stream(websocket: WebSocket):
    """جایگاه آینده برای جریان صوت به متن."""
    await websocket.accept()
    await websocket.send_json(
        {"type": "ready", "message": "کانال صوت به متن آماده است و به‌زودی فعال می‌شود."}
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await websocket.send_json(
                    {"type": "audio_ack", "note": "تبدیل گفتار به متن هنوز فعال نیست."}
                )
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
            elif t == "close":
                await websocket.close()
                break
            else:
                await websocket.send_json({"type": "info", "message": "فقط پیام‌های ساده پشتیبانی می‌شود."})
    except WebSocketDisconnect:
        return


@app.get("/api/health")
async def health():
    agent = get_agent()
    ollama_ok = False
    try:
        await ollama_client.check_ollama()
        ollama_ok = True
    except OllamaError:
        pass
    return {
        "status": "ok",
        "name": APP_TITLE,
        "chunks": agent.document_count,
        "ollama": ollama_ok,
        "voip": "/ws/voip-stream",
    }


@app.get("/")
async def index():
    path = FRONTEND_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="رابط کاربری پیدا نشد.")
    return FileResponse(path)


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

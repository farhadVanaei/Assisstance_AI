"""In-memory RAG store + Ollama chat with clean Persian answers and offline fallback."""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ollama_rag import client as ollama_client
from ollama_rag.client import OllamaError
from ollama_rag.text_clean import clean_text, sanitize_for_prompt

_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077FA-Za-z0-9]+")

# Fragments that mean the model echoed the prompt / leaked scaffolding
_LEAK_MARKERS = [
    "زمینهٔ دانش",
    "زمینه دانش",
    "متن دانش",
    "متن زیر",
    "پرسش کاربر",
    "فقط برای پاسخ",
    "در خروجی تکرار نکن",
    "منبع:",
    "system:",
    "user:",
    "assistant:",
    "context:",
    "rag:",
    "دانسته‌های مرتبط",
]

# Exact Persian system role required by product spec
SYSTEM_PROMPT = (
    "تو جواد هستی، یک دستیار هوشمند و محلی فارسی‌زبان. "
    "متن ارائه شده را بخوان و به پرسش کاربر به صورت طبیعی، کوتاه و به زبان فارسی روان پاسخ بده. "
    "از تکرار متن، نمایش لینک یا کدهای اضافی خودداری کن."
)


@dataclass
class Chunk:
    source: str
    text: str
    kind: str = "text"


@dataclass
class JavadOllamaAgent:
    chunks: List[Chunk] = field(default_factory=list)
    histories: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def clear(self) -> None:
        with self._lock:
            self.chunks.clear()
            self.histories.clear()

    @property
    def document_count(self) -> int:
        return len(self.chunks)

    def ingest(self, text: str, source: str, kind: str = "text") -> int:
        text = clean_text(text or "")
        if len(text) < 20:
            return 0
        parts = _split_chunks(text, max_len=600)
        added = 0
        with self._lock:
            for part in parts:
                self.chunks.append(Chunk(source=source or "سند", text=part, kind=kind))
                added += 1
        return added

    def _retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        if not self.chunks:
            return []
        q_tokens = set(_tokenize(query))
        scored: List[Tuple[float, Chunk]] = []
        for chunk in self.chunks:
            overlap = 0.0
            if q_tokens:
                c_tokens = set(_tokenize(chunk.text))
                overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            # substring / acronym boost (e.g. SPOT)
            q_low = query.lower().strip()
            if q_low and q_low in chunk.text.lower():
                overlap += 0.8
            if q_low and q_low in (chunk.source or "").lower():
                overlap += 0.5
            boost = _hash_sim(query, chunk.text)
            scored.append((overlap + 0.25 * boost, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        # If nothing matched, still return top chunks so model has some context
        return [c for _, c in scored[:top_k]]

    def _history(self, session_id: str) -> List[Dict[str, str]]:
        return self.histories.setdefault(session_id, [])

    async def chat(self, prompt: str, session_id: str = "default") -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return {
                "answer": "لطفاً پیام خود را به فارسی بنویسید.",
                "sources": [],
                "context_used": False,
                "chunks_used": 0,
                "model": ollama_client.CHAT_MODEL,
                "mode": "empty",
            }

        # Small-talk should not pull unrelated documents into the reply
        if _is_chitchat(prompt):
            retrieved = []
        else:
            retrieved = self._retrieve(prompt)

        sources = list(dict.fromkeys(c.source for c in retrieved if c.source))
        context_blob = "\n".join(c.text for c in retrieved)

        # Without knowledge: only small-talk goes to the model; facts use honest fallback
        if not retrieved and not _is_chitchat(prompt):
            answer = _fallback_answer(prompt, [])
            history = self._history(session_id)
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": answer})
            if len(history) > 12:
                self.histories[session_id] = history[-12:]
            return {
                "answer": answer,
                "sources": [],
                "context_used": False,
                "chunks_used": 0,
                "model": ollama_client.CHAT_MODEL,
                "mode": "fallback",
            }

        answer = ""
        mode = "ollama"
        try:
            answer = await self._ollama_answer(prompt, context_blob, session_id)
            answer = _sanitize_answer(answer, prompt)
            bad = not _is_good_answer(answer, prompt)
            if retrieved and answer and not _answer_grounded(answer, retrieved):
                bad = True
            if retrieved and answer and _has_foreign_junk(answer, retrieved, prompt):
                bad = True
            if retrieved and answer and _too_much_invention(answer, retrieved):
                bad = True
            if bad:
                mode = "fallback"
                answer = _fallback_answer(prompt, retrieved)
        except OllamaError:
            mode = "fallback"
            answer = _fallback_answer(prompt, retrieved)
        except Exception:
            mode = "fallback"
            answer = _fallback_answer(prompt, retrieved)

        answer = _sanitize_answer(answer, prompt)
        if not answer:
            answer = _fallback_answer(prompt, retrieved)

        history = self._history(session_id)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": answer})
        if len(history) > 12:
            self.histories[session_id] = history[-12:]

        return {
            "answer": answer,
            "sources": sources if retrieved else [],
            "context_used": bool(retrieved),
            "chunks_used": len(retrieved),
            "model": ollama_client.CHAT_MODEL,
            "mode": mode,
        }

    async def _ollama_answer(
        self, prompt: str, context_blob: str, session_id: str
    ) -> str:
        """Call Ollama: strict Persian system role + clean knowledge + user question only."""
        clean_ctx = sanitize_for_prompt(context_blob, max_chars=5000)
        system = SYSTEM_PROMPT
        if clean_ctx:
            system = (
                f"{SYSTEM_PROMPT}\n\n"
                f"متن ارائه شده:\n{clean_ctx}"
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for turn in self._history(session_id)[-4:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            if role == "assistant" and not _is_good_answer(content, ""):
                continue
            # Never re-inject long RAG scaffolding from history
            if len(content) > 600:
                continue
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt.strip()})
        return await ollama_client.chat(messages, temperature=0.25)


def _is_chitchat(prompt: str) -> bool:
    p = clean_text(prompt).lower()
    if len(p) > 48:
        return False
    cues = ("سلام", "درود", "خوبی", "حالت", "ممنون", "مرسی", "خداحافظ", "صبح بخیر", "شب خوش")
    return any(c in p for c in cues)


def _fallback_answer(prompt: str, chunks: List[Chunk]) -> str:
    """Offline conversational Persian reply from retrieved chunks (no Ollama)."""
    if not chunks:
        return (
            "هنوز چیزی در این باره در دانش من نیست. "
            "اگر سند یا وب‌سایتی دارید، اول در بخش پایگاه دانش اضافه کنید؛ "
            "بعد دوباره بپرسید."
        )

    # Prefer chunk that mentions the query term
    q = prompt.strip()
    best = chunks[0]
    for c in chunks:
        if q.lower() in c.text.lower() or q.lower() in (c.source or "").lower():
            best = c
            break

    fact = _first_sentences(best.text, max_chars=280)
    title = clean_text(best.source or "")
    # Don't speak raw filenames in the chat reply
    if re.search(r"\.(txt|pdf|html?|md|docx?)$", title, re.I) or re.fullmatch(r"[a-f0-9]{8,}_.+", title):
        title = ""
    if not fact:
        fact = best.text[:220].rstrip() + "…"

    # Natural spoken framing — never dump URLs or labels
    if title and len(title) < 80 and title.lower() not in fact.lower():
        return f"طبق چیزی که یاد گرفته‌ام، دربارهٔ «{title}» می‌توانم بگویم: {fact}"
    return f"طبق چیزی که یاد گرفته‌ام، {fact}"


def _first_sentences(text: str, max_chars: int = 280) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?؟])\s+", text)
    out: List[str] = []
    total = 0
    for p in parts:
        p = p.strip()
        if len(p) < 15:
            continue
        if any(m in p for m in _LEAK_MARKERS):
            continue
        out.append(p)
        total += len(p)
        if total >= max_chars or len(out) >= 2:
            break
    joined = " ".join(out).strip()
    if not joined:
        joined = text[:max_chars].rstrip()
    if len(joined) > max_chars:
        joined = joined[:max_chars].rsplit(" ", 1)[0] + "…"
    if joined and joined[-1] not in ".!?؟…":
        joined += "."
    return joined


def _sanitize_answer(answer: str, prompt: str) -> str:
    if not answer:
        return ""
    text = str(answer).strip()
    # Drop common leaked prefixes/blocks
    text = re.sub(
        r"(?is)^\s*(assistant|جواد)\s*[:：\-–]\s*",
        "",
        text,
    )
    for marker in _LEAK_MARKERS:
        if marker in text:
            # Keep only the part after the last marker line if possible
            lines = [
                ln
                for ln in text.splitlines()
                if marker not in ln and not ln.strip().startswith("http")
            ]
            text = " ".join(lines).strip() or text

    text = clean_text(text)
    # Remove accidental echo of the full user prompt
    if prompt and text.strip() == prompt.strip():
        return ""
    # Strip leading "پرسش: ..." style leftovers
    text = re.sub(r"(?is)^\s*پرسش\s*[:：].{0,80}\n?", "", text).strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_good_answer(answer: str, prompt: str) -> bool:
    if not answer or len(answer) < 8:
        return False
    low = answer.lower()
    if any(m.lower() in low for m in _LEAK_MARKERS):
        return False
    if prompt and prompt.strip() and answer.strip() == prompt.strip():
        return False
    persian = len(re.findall(r"[\u0600-\u06FF]", answer))
    latin_words = re.findall(r"[A-Za-z]{4,}", answer)
    # Allow short Latin tokens like SPOT/API inside otherwise Persian answers
    if len(latin_words) >= 2 and persian < 25:
        return False
    if persian == 0 and len(latin_words) >= 1:
        return False
    if answer.count("طبق چیزی که یاد گرفته‌ام") > 1:
        return False
    # Reject obvious English leftovers mixed into Persian
    banned_en = ("instantaneous", "blessing", "click here", "http", "context", "prompt")
    if any(b in low for b in banned_en):
        return False
    return True


def _answer_grounded(answer: str, chunks: List[Chunk]) -> bool:
    """True if answer shares meaningful tokens with retrieved knowledge."""
    if not chunks or not answer:
        return False
    a_tok = set(_tokenize(answer))
    # drop ultra-common Persian stop-ish tokens
    stop = {"که", "را", "به", "از", "در", "این", "آن", "و", "یا", "یک", "است", "هست"}
    a_tok -= stop
    if not a_tok:
        return False
    pool: set[str] = set()
    for c in chunks:
        pool |= set(_tokenize(c.text)) | set(_tokenize(c.source))
    pool -= stop
    if not pool:
        return False
    overlap = a_tok & pool
    return len(overlap) >= 2 or (len(overlap) >= 1 and len(a_tok) <= 6)


def _has_foreign_junk(answer: str, chunks: List[Chunk], prompt: str) -> bool:
    """Reject answers that invent English words not present in knowledge/query."""
    allowed = set(re.findall(r"[A-Za-z0-9]{2,}", prompt or ""))
    for c in chunks:
        allowed |= set(re.findall(r"[A-Za-z0-9]{2,}", f"{c.source} {c.text}"))
    allowed_l = {a.lower() for a in allowed}
    for w in re.findall(r"[A-Za-z]{4,}", answer or ""):
        if w.lower() not in allowed_l:
            return True
    return False


def _too_much_invention(answer: str, chunks: List[Chunk]) -> bool:
    """If most content tokens are not in the knowledge, treat as hallucination."""
    stop = {"که", "را", "به", "از", "در", "این", "آن", "و", "یا", "یک", "است", "هست", "می", "شود"}
    a_tok = set(_tokenize(answer)) - stop
    if len(a_tok) < 4:
        return False
    pool: set[str] = set()
    for c in chunks:
        pool |= set(_tokenize(c.text)) | set(_tokenize(c.source))
    pool -= stop
    if not pool:
        return True
    unique = a_tok - pool
    return len(unique) > max(3, len(a_tok) // 2)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(clean_text(text or "")) if len(t) > 1]


def _split_chunks(text: str, max_len: int = 600) -> List[str]:
    if len(text) <= max_len:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        start = end
    return out


def _hash_sim(a: str, b: str) -> float:
    va = _hash_vec(a)
    vb = _hash_vec(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _hash_vec(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n:
        vec /= n
    return vec


_agent: Optional[JavadOllamaAgent] = None


def get_agent() -> JavadOllamaAgent:
    global _agent
    if _agent is None:
        _agent = JavadOllamaAgent()
    return _agent

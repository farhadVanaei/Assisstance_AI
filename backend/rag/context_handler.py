"""In-memory document store and Persian RAG context handler."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class DocumentChunk:
    source: str
    text: str
    kind: str = "text"


@dataclass
class ContextStore:
    chunks: List[DocumentChunk] = field(default_factory=list)

    def clear(self) -> None:
        self.chunks.clear()

    def add(self, source: str, text: str, kind: str = "text") -> int:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return 0
        parts = _split_chunks(cleaned, max_len=500)
        for part in parts:
            self.chunks.append(DocumentChunk(source=source, text=part, kind=kind))
        return len(parts)

    def retrieve(self, query: str, top_k: int = 4) -> List[DocumentChunk]:
        if not self.chunks:
            return []
        tokens = set(_tokenize(query))
        if not tokens:
            return self.chunks[:top_k]

        scored: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks:
            chunk_tokens = set(_tokenize(chunk.text))
            overlap = len(tokens & chunk_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(tokens), 1)
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return self.chunks[:top_k]
        return [item[1] for item in scored[:top_k]]

    def answer(self, prompt: str) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            return {
                "answer": "لطفاً پرسش خود را به فارسی بنویسید.",
                "sources": [],
                "context_used": False,
            }

        retrieved = self.retrieve(prompt)
        if not retrieved:
            return {
                "answer": (
                    "هنوز مدرکی در پایگاه دانش جواد بارگذاری نشده است. "
                    "لطفاً از بخش آپلود، فایل یا آدرس وبسایت را اضافه کنید."
                ),
                "sources": [],
                "context_used": False,
            }

        context_block = "\n\n".join(
            f"[{i + 1}] ({c.source})\n{c.text}" for i, c in enumerate(retrieved)
        )
        answer = _compose_persian_answer(prompt, retrieved, context_block)
        sources = list(dict.fromkeys(c.source for c in retrieved))
        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "chunks_used": len(retrieved),
        }


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[\w\u0600-\u06FF]+", text.lower()) if len(t) > 1]


def _split_chunks(text: str, max_len: int = 500) -> List[str]:
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _compose_persian_answer(
    prompt: str, chunks: List[DocumentChunk], context_block: str
) -> str:
    """Compose a grounded Persian reply from retrieved context (no external LLM)."""
    snippets = []
    for i, chunk in enumerate(chunks[:3], start=1):
        excerpt = chunk.text[:280]
        if len(chunk.text) > 280:
            excerpt += "…"
        snippets.append(f"{i}. از «{chunk.source}»: {excerpt}")

    body = "\n".join(snippets)
    return (
        f"بر اساس محتوای بارگذاری‌شده در پایگاه دانش جواد، پاسخ به پرسش شما "
        f"(«{prompt}») به این شکل است:\n\n"
        f"{body}\n\n"
        "اگر نیاز به جزئیات بیشتری دارید، پرسش را دقیق‌تر مطرح کنید "
        "یا مدرک بیشتری آپلود نمایید."
    )


store = ContextStore()

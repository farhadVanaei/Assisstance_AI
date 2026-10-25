"""Persistent RAG store + hybrid retrieval + Ollama chat with offline fallback."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ollama_rag import client as ollama_client
from ollama_rag.client import OllamaError
from ollama_rag.config import (
    CHUNK_MAX_LEN,
    CHUNK_OVERLAP,
    DEFAULT_LANGUAGE,
    EMBED_MODEL,
    RELEVANCE_THRESHOLD,
    RETRIEVAL_CANDIDATES,
    RETRIEVAL_TOP_K,
)
from ollama_rag.languages import (
    all_chitchat_cues,
    detect_language,
    empty_prompt,
    fallback_prefix,
    leak_markers,
    no_knowledge,
    normalize_lang,
    system_prompt,
)
from ollama_rag.metrics import metrics, timed
from ollama_rag.store import SQLiteStore, get_store
from ollama_rag.text_clean import clean_text, sanitize_for_prompt
from ollama_rag.vector_store import VectorStore, get_vector_store, hash_embed

logger = logging.getLogger("javad.agent")

_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077FA-Za-z0-9]+")

_STOPWORDS = {
    "\u06A9\u0647", "\u0631\u0627", "\u0628\u0647", "\u0627\u0632", "\u062F\u0631", "\u0627\u06CC\u0646", "\u0622\u0646", "\u0648", "\u06CC\u0627", "\u06CC\u06A9", "\u0627\u0633\u062A", "\u0647\u0633\u062A",
    "\u0645\u06CC", "\u0634\u0648\u062F", "\u0628\u0627", "\u0628\u0631\u0627\u06CC", "\u062A\u0627", "\u0647\u0645", "\u0628\u0631", "\u0647\u0627", "\u0647\u0627\u06CC", "\u06A9\u0646\u06CC\u062F", "\u06A9\u0646\u0645",
    "\u0686\u06CC\u0633\u062A", "\u0686\u0647", "\u0686\u06CC", "\u06A9\u062F\u0627\u0645", "\u0622\u06CC\u0627", "\u0628\u0639\u062F", "\u067E\u0633", "\u0627\u06AF\u0631", "\u0628\u0627\u06CC\u062F", "\u0628\u0648\u062F",
    "the", "and", "of", "to", "a", "is", "in", "for",
}


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    kind: str = "text"
    source_id: str = ""
    score: float = 0.0
    url: str = ""
    images: List[str] = field(default_factory=list)


@dataclass
class JavadOllamaAgent:
    user_id: str = "default"
    store: SQLiteStore = field(default_factory=get_store)
    vectors: VectorStore = field(default_factory=get_vector_store)
    language: str = DEFAULT_LANGUAGE
    model: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _chunk_cache: Dict[str, Chunk] = field(default_factory=dict)
    _bm25_corpus: List[str] = field(default_factory=list)
    _bm25_ids: List[str] = field(default_factory=list)
    _bm25 = None
    _embed_model_used: str = ""
    _corpus_common_terms: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.reload_cache()

    def reload_cache(self) -> None:
        """Load chunks from SQLite into memory cache and rebuild BM25."""
        records = self.store.load_chunks(self.user_id)
        source_meta: Dict[str, Dict[str, Any]] = {
            s.id: (s.meta or {}) for s in self.store.list_sources(self.user_id)
        }
        cache: Dict[str, Chunk] = {}
        corpus: List[str] = []
        ids: List[str] = []
        for r in records:
            meta = source_meta.get(r.source_id, {})
            cache[r.id] = Chunk(
                id=r.id,
                source=r.source,
                text=r.text,
                kind=r.kind,
                source_id=r.source_id,
                url=str(meta.get("url") or ""),
                images=list(meta.get("images") or []),
            )
            corpus.append(r.text)
            ids.append(r.id)
        with self._lock:
            self._chunk_cache = cache
            self._bm25_corpus = corpus
            self._bm25_ids = ids
            self._bm25 = _build_bm25(corpus)
            self._corpus_common_terms = _detect_common_terms(corpus)
        logger.info("Loaded %d chunks for user=%s", len(cache), self.user_id)

    def clear(self) -> None:
        with self._lock:
            ids = list(self._chunk_cache.keys())
            self._chunk_cache.clear()
            self._bm25_corpus.clear()
            self._bm25_ids.clear()
            self._bm25 = None
            self._corpus_common_terms = set()
        self.vectors.delete(ids)
        self.vectors.delete_by_user(self.user_id)
        self.store.clear(self.user_id)

    @property
    def document_count(self) -> int:
        return len(self._chunk_cache)

    def list_sources(self) -> List[Dict[str, Any]]:
        result = []
        for s in self.store.list_sources(self.user_id):
            meta = s.meta or {}
            result.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "kind": s.kind,
                    "chunk_count": s.chunk_count,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "expires_at": s.expires_at,
                    "url": str(meta.get("url") or ""),
                    "images": list(meta.get("images") or []),
                }
            )
        return result

    def delete_source(self, source_id: str) -> Dict[str, Any]:
        src = self.store.get_source(source_id, self.user_id)
        if not src:
            return {"ok": False, "deleted_chunks": 0}
        # Collect chunk ids before delete
        to_remove = [
            cid for cid, c in self._chunk_cache.items() if c.source_id == source_id
        ]
        deleted = self.store.delete_source(source_id, self.user_id)
        self.vectors.delete(to_remove)
        self.vectors.delete_by_source(source_id)
        self.reload_cache()
        return {"ok": True, "deleted_chunks": deleted, "source": src.name}

    def ingest(
        self,
        text: str,
        source: str,
        kind: str = "text",
        *,
        ttl_days: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        with timed() as t:
            added = self._ingest_impl(text, source, kind, ttl_days=ttl_days, meta=meta)
        metrics.record_ingest(t.elapsed)
        return added

    def _ingest_impl(
        self,
        text: str,
        source: str,
        kind: str = "text",
        *,
        ttl_days: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        text = clean_text(text or "")
        if len(text) < 20:
            return 0

        src = self.store.add_source(
            name=source or "سند",
            kind=kind,
            text=text,
            user_id=self.user_id,
            meta=meta,
            ttl_days=ttl_days,
        )
        if src is None:
            logger.info("Duplicate content skipped: %s", source)
            return 0

        parts = _split_chunks(text, max_len=CHUNK_MAX_LEN, overlap=CHUNK_OVERLAP)
        records = self.store.add_chunks(
            source_id=src.id,
            source=src.name,
            kind=kind,
            texts=parts,
            user_id=self.user_id,
            expires_at=src.expires_at,
        )
        if not records:
            return 0

        embeddings = self._embed_texts([r.text for r in records])
        self.vectors.upsert(
            ids=[r.id for r in records],
            embeddings=embeddings,
            documents=[r.text for r in records],
            metadatas=[
                {
                    "source": r.source,
                    "source_id": r.source_id,
                    "kind": r.kind,
                    "user_id": self.user_id,
                }
                for r in records
            ],
        )
        self.reload_cache()
        return len(records)

    def _fallback_dim(self) -> int:
        """Match the collection's existing embedding dimension so a transient Ollama
        outage never produces a vector of the wrong size for the active collection."""
        return self.vectors.expected_dim() or 384

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = ollama_client.embed_sync(texts, model=EMBED_MODEL)
        if vectors and len(vectors) == len(texts):
            self._embed_model_used = EMBED_MODEL
            return vectors
        logger.warning("Ollama embeddings unavailable; using hash fallback")
        self._embed_model_used = "hash-fallback"
        dim = self._fallback_dim()
        return [hash_embed(t, dim=dim) for t in texts]

    async def _embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        vectors = await ollama_client.embed(texts, model=EMBED_MODEL)
        if vectors and len(vectors) == len(texts):
            self._embed_model_used = EMBED_MODEL
            return vectors
        self._embed_model_used = "hash-fallback"
        dim = self._fallback_dim()
        return [hash_embed(t, dim=dim) for t in texts]

    async def reindex(self) -> Dict[str, Any]:
        """Re-embed all chunks (e.g. after embedding model change)."""
        records = self.store.load_chunks(self.user_id)
        if not records:
            return {"reindexed": 0, "model": EMBED_MODEL}
        batch_size = 32
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            embeddings = await self._embed_texts_async([r.text for r in batch])
            self.vectors.upsert(
                ids=[r.id for r in batch],
                embeddings=embeddings,
                documents=[r.text for r in batch],
                metadatas=[
                    {
                        "source": r.source,
                        "source_id": r.source_id,
                        "kind": r.kind,
                        "user_id": self.user_id,
                    }
                    for r in batch
                ],
            )
            total += len(batch)
        self.reload_cache()
        return {"reindexed": total, "model": self._embed_model_used or EMBED_MODEL}

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Chunk]:
        if not self._chunk_cache:
            return []

        candidates = max(RETRIEVAL_CANDIDATES, top_k)
        # Vector scores (convert distance → similarity)
        q_emb = self._embed_texts([query])[0]
        vector_hits = self.vectors.query(q_emb, n_results=candidates, user_id=self.user_id)
        vec_scores: Dict[str, float] = {}
        for cid, dist in vector_hits:
            # cosine distance → similarity in [0, 1]
            sim = max(0.0, 1.0 - float(dist))
            vec_scores[cid] = sim

        # BM25 scores
        bm25_scores: Dict[str, float] = {}
        if self._bm25 is not None and self._bm25_ids:
            tokens = _tokenize(query)
            if tokens:
                try:
                    raw = self._bm25.get_scores(tokens)
                    max_raw = max(raw) if len(raw) else 1.0
                    max_raw = max_raw if max_raw > 0 else 1.0
                    for idx, score in enumerate(raw):
                        bm25_scores[self._bm25_ids[idx]] = float(score) / max_raw
                except Exception:
                    pass

        # Token overlap + substring boost
        q_tokens = set(_tokenize(query))
        q_content_tokens = q_tokens - _STOPWORDS - self._corpus_common_terms
        q_low = query.lower().strip()
        hybrid: Dict[str, float] = {}
        all_ids = set(vec_scores) | set(bm25_scores) | set(self._chunk_cache)
        for cid in all_ids:
            chunk = self._chunk_cache.get(cid)
            if not chunk:
                continue
            c_tokens = set(_tokenize(chunk.text))
            overlap = 0.0
            if q_tokens:
                overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            substring_hit = False
            if q_low and q_low in chunk.text.lower():
                overlap += 0.8
                substring_hit = True
            if q_low and q_low in (chunk.source or "").lower():
                overlap += 0.5
                substring_hit = True
            v = vec_scores.get(cid, 0.0)
            b = bm25_scores.get(cid, 0.0)
            # Hybrid merge
            score = 0.45 * v + 0.35 * b + 0.20 * min(1.0, overlap)
            # Hard lexical gate: on a dense, single-topic corpus, embedding similarity
            # alone does not reliably separate relevant from irrelevant chunks
            # (observed ~0.82-0.90 vector scores across topically unrelated documents
            # that merely share the domain, e.g. "bank"). Require the query to share
            # at least one real, corpus-discriminating word with the chunk —
            # otherwise a high vector/BM25 score is almost certainly a false
            # positive, so zero it out rather than cite an unrelated document as if
            # it answered the question. Short/generic queries (<2 content words)
            # skip this gate since there's not enough signal to safely filter on.
            if len(q_content_tokens) >= 2 and not substring_hit:
                shared = q_content_tokens & c_tokens
                source_tokens = set(_tokenize(chunk.source or "")) - _STOPWORDS - self._corpus_common_terms
                if not shared:
                    shared = q_content_tokens & source_tokens
                if not shared:
                    score = 0.0
            hybrid[cid] = score

        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)
        # Re-rank top candidates with simple cross-check
        top_candidates = ranked[:candidates]
        reranked = _rerank(query, top_candidates, self._chunk_cache)

        results: List[Chunk] = []
        for cid, score in reranked:
            if score < RELEVANCE_THRESHOLD:
                continue
            chunk = self._chunk_cache.get(cid)
            if not chunk:
                continue
            results.append(
                Chunk(
                    id=chunk.id,
                    source=chunk.source,
                    text=chunk.text,
                    kind=chunk.kind,
                    source_id=chunk.source_id,
                    score=score,
                    url=chunk.url,
                    images=chunk.images,
                )
            )
            if len(results) >= top_k:
                break

        # Deliberately do NOT force-return a below-threshold chunk when nothing
        # clears the bar: serving the single best-scoring-but-irrelevant chunk makes
        # the answer look like it addresses the question while actually ignoring it
        # (e.g. an unrelated article confidently presented as the answer). An honest
        # "I don't know" is better than a fluent non-sequitur.
        return results

    def _history(self, session_id: str) -> List[Dict[str, str]]:
        return self.store.get_history(session_id, self.user_id, limit=12)

    def _append_turn(self, session_id: str, prompt: str, answer: str) -> None:
        self.store.append_history(session_id, "user", prompt, self.user_id)
        self.store.append_history(session_id, "assistant", answer, self.user_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self.store.list_sessions(self.user_id)

    def rename_session(self, session_id: str, title: str) -> bool:
        return self.store.rename_session(session_id, title, self.user_id)

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete_session(session_id, self.user_id)

    async def chat(
        self,
        prompt: str,
        session_id: str = "default",
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        with timed() as t:
            result = await self._chat_impl(
                prompt, session_id, language=language, model=model
            )
        metrics.record_chat(t.elapsed, result.get("mode") or "unknown")
        return result

    async def _chat_impl(
        self,
        prompt: str,
        session_id: str = "default",
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        lang = normalize_lang(language or detect_language(prompt, self.language))
        chat_model = model or self.model

        if not prompt:
            return {
                "answer": empty_prompt(lang),
                "sources": [],
                "citations": [],
                "context_used": False,
                "chunks_used": 0,
                "model": chat_model or ollama_client.CHAT_MODEL,
                "mode": "empty",
                "language": lang,
            }

        if _is_chitchat(prompt, lang):
            retrieved: List[Chunk] = []
        else:
            retrieved = self._retrieve(prompt)

        sources = list(dict.fromkeys(c.source for c in retrieved if c.source))
        citations = [
            {
                "source": c.source,
                "source_id": c.source_id,
                "score": round(c.score, 4),
                "excerpt": c.text[:160] + ("…" if len(c.text) > 160 else ""),
                "url": c.url,
                "images": c.images,
            }
            for c in retrieved
        ]
        context_blob = "\n".join(c.text for c in retrieved)

        if not retrieved and not _is_chitchat(prompt, lang):
            answer = _fallback_answer(prompt, [], lang)
            self._append_turn(session_id, prompt, answer)
            return {
                "answer": answer,
                "sources": [],
                "citations": [],
                "context_used": False,
                "chunks_used": 0,
                "model": chat_model or ollama_client.CHAT_MODEL,
                "mode": "fallback",
                "language": lang,
            }

        answer = ""
        mode = "ollama"
        try:
            answer = await self._ollama_answer(
                prompt, context_blob, session_id, lang=lang, model=chat_model
            )
            answer = _sanitize_answer(answer, prompt, lang)
            bad = not _is_good_answer(answer, prompt, lang)
            if retrieved and answer and not _answer_grounded(answer, retrieved):
                bad = True
            if retrieved and answer and _has_foreign_junk(answer, retrieved, prompt):
                bad = True
            if retrieved and answer and _too_much_invention(answer, retrieved):
                bad = True
            if bad:
                mode = "fallback"
                answer = _fallback_answer(prompt, retrieved, lang)
        except OllamaError:
            mode = "fallback"
            answer = _fallback_answer(prompt, retrieved, lang)
        except Exception:
            mode = "fallback"
            answer = _fallback_answer(prompt, retrieved, lang)

        answer = _sanitize_answer(answer, prompt, lang)
        if not answer:
            answer = _fallback_answer(prompt, retrieved, lang)

        self._append_turn(session_id, prompt, answer)
        return {
            "answer": answer,
            "sources": sources if retrieved else [],
            "citations": citations if retrieved else [],
            "context_used": bool(retrieved),
            "chunks_used": len(retrieved),
            "model": chat_model or ollama_client.CHAT_MODEL,
            "mode": mode,
            "language": lang,
        }

    async def chat_stream(
        self,
        prompt: str,
        session_id: str = "default",
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield NDJSON-friendly events: meta → token* → done."""
        prompt = (prompt or "").strip()
        lang = normalize_lang(language or detect_language(prompt, self.language))
        chat_model = model or self.model

        if not prompt:
            yield {"type": "done", "answer": empty_prompt(lang), "mode": "empty", "sources": [], "citations": []}
            return

        if _is_chitchat(prompt, lang):
            retrieved: List[Chunk] = []
        else:
            retrieved = self._retrieve(prompt)

        sources = list(dict.fromkeys(c.source for c in retrieved if c.source))
        citations = [
            {
                "source": c.source,
                "source_id": c.source_id,
                "score": round(c.score, 4),
                "excerpt": c.text[:160] + ("…" if len(c.text) > 160 else ""),
                "url": c.url,
                "images": c.images,
            }
            for c in retrieved
        ]
        yield {
            "type": "meta",
            "sources": sources,
            "citations": citations,
            "chunks_used": len(retrieved),
            "language": lang,
            "model": chat_model or ollama_client.CHAT_MODEL,
        }

        if not retrieved and not _is_chitchat(prompt, lang):
            answer = _fallback_answer(prompt, [], lang)
            self._append_turn(session_id, prompt, answer)
            yield {"type": "token", "text": answer}
            yield {"type": "done", "answer": answer, "mode": "fallback", "sources": [], "citations": []}
            return

        context_blob = "\n".join(c.text for c in retrieved)
        pieces: List[str] = []
        mode = "ollama"
        try:
            messages = self._build_messages(prompt, context_blob, session_id, lang)
            async for token in ollama_client.chat_stream(
                messages, model=chat_model, temperature=0.25
            ):
                pieces.append(token)
                yield {"type": "token", "text": token}
            answer = _sanitize_answer("".join(pieces), prompt, lang)
            if not _is_good_answer(answer, prompt, lang):
                mode = "fallback"
                answer = _fallback_answer(prompt, retrieved, lang)
                yield {"type": "token", "text": answer, "replace": True}
        except Exception:
            mode = "fallback"
            answer = _fallback_answer(prompt, retrieved, lang)
            yield {"type": "token", "text": answer, "replace": True}

        self._append_turn(session_id, prompt, answer)
        yield {
            "type": "done",
            "answer": answer,
            "mode": mode,
            "sources": sources if retrieved else [],
            "citations": citations if retrieved else [],
            "context_used": bool(retrieved),
            "chunks_used": len(retrieved),
            "language": lang,
            "model": chat_model or ollama_client.CHAT_MODEL,
        }

    def _build_messages(
        self, prompt: str, context_blob: str, session_id: str, lang: str
    ) -> List[Dict[str, str]]:
        clean_ctx = sanitize_for_prompt(context_blob, max_chars=5000)
        system = system_prompt(lang)
        if clean_ctx:
            label = "متن ارائه شده" if lang == "fa" else "Provided text"
            system = f"{system}\n\n{label}:\n{clean_ctx}"
        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for turn in self._history(session_id)[-4:]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            if role == "assistant" and not _is_good_answer(content, "", lang):
                continue
            if len(content) > 600:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt.strip()})
        return messages

    async def _ollama_answer(
        self,
        prompt: str,
        context_blob: str,
        session_id: str,
        *,
        lang: str = "fa",
        model: Optional[str] = None,
    ) -> str:
        messages = self._build_messages(prompt, context_blob, session_id, lang)
        return await ollama_client.chat(messages, model=model, temperature=0.25)

    def status(self) -> Dict[str, Any]:
        return {
            "chunks": self.document_count,
            "sources": len(self.list_sources()),
            "vector_backend": self.vectors.backend,
            "vector_count": self.vectors.count(),
            "embed_model": self._embed_model_used or EMBED_MODEL,
            "language": self.language,
            "user_id": self.user_id,
        }


def _build_bm25(corpus: List[str]):
    if not corpus:
        return None
    try:
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(doc) for doc in corpus]
        return BM25Okapi(tokenized)
    except Exception:
        return None


def _detect_common_terms(corpus: List[str], doc_freq_ratio: float = 0.35, min_docs: int = 6) -> set:
    """Words appearing in a large fraction of chunks (e.g. a company's own name,
    or "bank" on a banking company's site) don't discriminate relevance for that
    corpus — sharing only such a word with a query is not a real relevance signal.
    Detected dynamically per corpus rather than hardcoded, so it adapts to whatever
    knowledge base is actually loaded instead of being tuned to one deployment."""
    if len(corpus) < min_docs:
        return set()
    doc_count: Dict[str, int] = {}
    for text in corpus:
        for tok in set(_tokenize(text)):
            doc_count[tok] = doc_count.get(tok, 0) + 1
    threshold = len(corpus) * doc_freq_ratio
    return {tok for tok, n in doc_count.items() if n >= threshold}


def _rerank(
    query: str,
    candidates: List[Tuple[str, float]],
    cache: Dict[str, Chunk],
) -> List[Tuple[str, float]]:
    """Lightweight re-ranking: boost chunks with query term density + title match."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return candidates
    q_set = set(q_tokens)
    rescored: List[Tuple[str, float]] = []
    for cid, score in candidates:
        chunk = cache.get(cid)
        if not chunk:
            continue
        c_tokens = _tokenize(chunk.text)
        if not c_tokens:
            rescored.append((cid, score))
            continue
        density = len(q_set & set(c_tokens)) / max(len(q_set), 1)
        title_boost = 0.15 if any(t in (chunk.source or "").lower() for t in q_tokens) else 0.0
        # Prefer shorter, denser chunks slightly
        length_penalty = min(0.1, len(chunk.text) / 20000)
        rescored.append((cid, score + 0.25 * density + title_boost - length_penalty))
    rescored.sort(key=lambda x: x[1], reverse=True)
    return rescored


def _is_chitchat(prompt: str, lang: str = "fa") -> bool:
    p = clean_text(prompt).lower()
    if len(p) > 48:
        return False
    cues = all_chitchat_cues()
    return any(c in p for c in cues)


def _fallback_answer(prompt: str, chunks: List[Chunk], lang: str = "fa") -> str:
    if not chunks:
        return no_knowledge(lang)

    q = prompt.strip()
    best = chunks[0]
    for c in chunks:
        if q.lower() in c.text.lower() or q.lower() in (c.source or "").lower():
            best = c
            break

    fact = _first_sentences(best.text, max_chars=280, lang=lang)
    title = clean_text(best.source or "")
    if re.search(r"\.(txt|pdf|html?|md|docx?)$", title, re.I) or re.fullmatch(
        r"[a-f0-9]{8,}_.+", title
    ):
        title = ""
    if not fact:
        fact = best.text[:220].rstrip() + "…"

    prefix = fallback_prefix(lang)
    if title and len(title) < 80 and title.lower() not in fact.lower():
        if lang == "fa":
            return f"{prefix} دربارهٔ «{title}» می‌توانم بگویم: {fact}"
        if lang == "ar":
            return f"{prefix} عن «{title}»: {fact}"
        return f"{prefix} about «{title}»: {fact}"
    return f"{prefix} {fact}"


def _first_sentences(text: str, max_chars: int = 280, lang: str = "fa") -> str:
    text = clean_text(text)
    if not text:
        return ""
    markers = leak_markers(lang)
    parts = re.split(r"(?<=[.!?؟])\s+", text)
    out: List[str] = []
    total = 0
    for p in parts:
        p = p.strip()
        if len(p) < 15:
            continue
        if any(m in p for m in markers):
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


def _sanitize_answer(answer: str, prompt: str, lang: str = "fa") -> str:
    if not answer:
        return ""
    text = str(answer).strip()
    text = re.sub(r"(?is)^\s*(assistant|جواد|javad)\s*[:：\-–]\s*", "", text)
    for marker in leak_markers(lang):
        if marker in text:
            lines = [
                ln
                for ln in text.splitlines()
                if marker not in ln and not ln.strip().startswith("http")
            ]
            text = " ".join(lines).strip() or text
    text = clean_text(text)
    if prompt and text.strip() == prompt.strip():
        return ""
    text = re.sub(r"(?is)^\s*پرسش\s*[:：].{0,80}\n?", "", text).strip()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_good_answer(answer: str, prompt: str, lang: str = "fa") -> bool:
    if not answer or len(answer) < 8:
        return False
    low = answer.lower()
    if any(m.lower() in low for m in leak_markers(lang)):
        return False
    if prompt and prompt.strip() and answer.strip() == prompt.strip():
        return False
    persian = len(re.findall(r"[\u0600-\u06FF]", answer))
    latin_words = re.findall(r"[A-Za-z]{4,}", answer)
    if lang == "fa":
        if len(latin_words) >= 2 and persian < 25:
            return False
        if persian == 0 and len(latin_words) >= 1:
            return False
    if answer.count(fallback_prefix(lang)) > 1:
        return False
    banned_en = ("instantaneous", "blessing", "click here", "http", "context", "prompt")
    if any(b in low for b in banned_en):
        return False
    return True


def _answer_grounded(answer: str, chunks: List[Chunk]) -> bool:
    if not chunks or not answer:
        return False
    a_tok = set(_tokenize(answer))
    stop = {"که", "را", "به", "از", "در", "این", "آن", "و", "یا", "یک", "است", "هست", "the", "and", "of", "to", "a"}
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
    allowed = set(re.findall(r"[A-Za-z0-9]{2,}", prompt or ""))
    for c in chunks:
        allowed |= set(re.findall(r"[A-Za-z0-9]{2,}", f"{c.source} {c.text}"))
    allowed_l = {a.lower() for a in allowed}
    for w in re.findall(r"[A-Za-z]{4,}", answer or ""):
        if w.lower() not in allowed_l:
            return True
    return False


def _too_much_invention(answer: str, chunks: List[Chunk]) -> bool:
    stop = {
        "که", "را", "به", "از", "در", "این", "آن", "و", "یا", "یک", "است", "هست", "می", "شود",
        "the", "and", "of", "to", "a", "is", "in",
    }
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


def _split_chunks(text: str, max_len: int = 600, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks so boundary facts aren't lost."""
    if len(text) <= max_len:
        return [text]
    overlap = max(0, min(overlap, max_len // 2))
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
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


# --- agent registry (per-user) ---

_agents: Dict[str, JavadOllamaAgent] = {}
_agents_lock = threading.Lock()


def get_agent(user_id: str = "default") -> JavadOllamaAgent:
    uid = (user_id or "default").strip() or "default"
    with _agents_lock:
        if uid not in _agents:
            _agents[uid] = JavadOllamaAgent(user_id=uid)
        return _agents[uid]


def reset_agents() -> None:
    with _agents_lock:
        _agents.clear()

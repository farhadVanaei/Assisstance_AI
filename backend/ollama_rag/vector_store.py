"""Local ChromaDB vector store for chunk embeddings."""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ollama_rag.config import CHROMA_COLLECTION, CHROMA_PATH

logger = logging.getLogger("javad.vector_store")


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic bag-of-words hash embedding used when Ollama embed is unavailable."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t.lower() for t in text.split() if len(t) > 1]
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n:
        vec /= n
    return vec.tolist()


class VectorStore:
    """Chroma-backed vector index with in-process cosine fallback."""

    def __init__(
        self,
        path: Path | str | None = None,
        collection: str = CHROMA_COLLECTION,
    ) -> None:
        self.path = Path(path or CHROMA_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self._lock = threading.RLock()
        self._client = None
        self._col = None
        # Fallback in-memory index when chromadb is missing
        self._fallback: Dict[str, Tuple[List[float], Dict]] = {}
        self._use_chroma = False
        self._init()

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=str(self.path),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._col = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
            logger.info("ChromaDB ready at %s (%s)", self.path, self.collection_name)
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s); using in-memory cosine index", exc)
            self._use_chroma = False

    @property
    def backend(self) -> str:
        return "chromadb" if self._use_chroma else "memory"

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
    ) -> None:
        if not ids:
            return
        with self._lock:
            if self._use_chroma and self._col is not None:
                self._col.upsert(
                    ids=list(ids),
                    embeddings=[list(e) for e in embeddings],
                    documents=list(documents),
                    metadatas=[{k: ("" if v is None else v) for k, v in m.items()} for m in metadatas],
                )
            else:
                for i, eid in enumerate(ids):
                    self._fallback[eid] = (list(embeddings[i]), dict(metadatas[i]))

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self._lock:
            if self._use_chroma and self._col is not None:
                try:
                    self._col.delete(ids=list(ids))
                except Exception:
                    pass
            for eid in ids:
                self._fallback.pop(eid, None)

    def delete_by_source(self, source_id: str) -> None:
        with self._lock:
            if self._use_chroma and self._col is not None:
                try:
                    self._col.delete(where={"source_id": source_id})
                except Exception:
                    pass
            to_drop = [
                eid
                for eid, (_e, meta) in self._fallback.items()
                if meta.get("source_id") == source_id
            ]
            for eid in to_drop:
                self._fallback.pop(eid, None)

    def delete_by_user(self, user_id: str) -> None:
        with self._lock:
            if self._use_chroma and self._col is not None:
                try:
                    self._col.delete(where={"user_id": user_id})
                except Exception:
                    pass
            to_drop = [
                eid
                for eid, (_e, meta) in self._fallback.items()
                if meta.get("user_id") == user_id
            ]
            for eid in to_drop:
                self._fallback.pop(eid, None)

    def clear(self) -> None:
        with self._lock:
            if self._use_chroma and self._client is not None:
                try:
                    self._client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self._col = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            self._fallback.clear()

    def query(
        self,
        embedding: Sequence[float],
        n_results: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Return list of (chunk_id, distance) sorted by ascending distance."""
        with self._lock:
            if self._use_chroma and self._col is not None:
                kwargs: Dict = {
                    "query_embeddings": [list(embedding)],
                    "n_results": max(1, n_results),
                    "include": ["distances"],
                }
                if user_id:
                    kwargs["where"] = {"user_id": user_id}
                try:
                    result = self._col.query(**kwargs)
                except Exception:
                    return []
                ids = (result.get("ids") or [[]])[0]
                dists = (result.get("distances") or [[]])[0]
                return list(zip(ids, [float(d) for d in dists]))

            # Cosine distance fallback
            q = np.asarray(embedding, dtype=np.float32)
            qn = float(np.linalg.norm(q)) or 1.0
            scored: List[Tuple[str, float]] = []
            for eid, (vec, meta) in self._fallback.items():
                if user_id and meta.get("user_id") != user_id:
                    continue
                v = np.asarray(vec, dtype=np.float32)
                sim = float(np.dot(q, v) / (qn * (float(np.linalg.norm(v)) or 1.0)))
                scored.append((eid, 1.0 - sim))  # distance
            scored.sort(key=lambda x: x[1])
            return scored[:n_results]

    def count(self) -> int:
        with self._lock:
            if self._use_chroma and self._col is not None:
                try:
                    return int(self._col.count())
                except Exception:
                    return 0
            return len(self._fallback)


_vs: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore()
    return _vs


def hash_embed(text: str, dim: int = 384) -> List[float]:
    return _hash_embed(text, dim=dim)

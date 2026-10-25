"""SQLite persistence for chunks, sources, and chat history."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ollama_rag.config import KNOWLEDGE_TTL_DAYS, SQLITE_PATH


@dataclass
class ChunkRecord:
    id: str
    source_id: str
    source: str
    text: str
    kind: str
    content_hash: str
    user_id: str
    created_at: float
    expires_at: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class SourceRecord:
    id: str
    name: str
    kind: str
    content_hash: str
    user_id: str
    chunk_count: int
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class SQLiteStore:
    """Thread-safe SQLite store for knowledge metadata and chat history."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or SQLITE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    meta TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    meta TEXT,
                    FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    title TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (id, user_id)
                );
                CREATE TABLE IF NOT EXISTS crawl_pages (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    title TEXT,
                    last_crawled REAL NOT NULL,
                    user_id TEXT NOT NULL DEFAULT 'default'
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
                CREATE INDEX IF NOT EXISTS idx_sources_user ON sources(user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, user_id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at);
                """
            )

    def purge_expired(self, user_id: str = "default") -> int:
        now = time.time()
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "SELECT id FROM sources WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
                (user_id, now),
            )
            expired = [r["id"] for r in cur.fetchall()]
            for sid in expired:
                conn.execute("DELETE FROM sources WHERE id = ?", (sid,))
            conn.execute(
                "DELETE FROM chunks WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at < ?",
                (user_id, now),
            )
            return len(expired)

    def add_source(
        self,
        name: str,
        kind: str,
        text: str,
        user_id: str = "default",
        meta: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
    ) -> Optional[SourceRecord]:
        """Register a source. Returns None if duplicate content hash for this user."""
        h = content_hash(text)
        now = time.time()
        ttl = KNOWLEDGE_TTL_DAYS if ttl_days is None else ttl_days
        expires = (now + ttl * 86400) if ttl and ttl > 0 else None

        with self._lock, self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM sources WHERE user_id = ? AND content_hash = ?",
                (user_id, h),
            ).fetchone()
            if existing:
                return None

            sid = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO sources (id, name, kind, content_hash, user_id, chunk_count,
                                    created_at, updated_at, expires_at, meta)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    sid,
                    name,
                    kind,
                    h,
                    user_id,
                    now,
                    now,
                    expires,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            return SourceRecord(
                id=sid,
                name=name,
                kind=kind,
                content_hash=h,
                user_id=user_id,
                chunk_count=0,
                created_at=now,
                updated_at=now,
                expires_at=expires,
                meta=meta or {},
            )

    def add_chunks(
        self,
        source_id: str,
        source: str,
        kind: str,
        texts: List[str],
        user_id: str = "default",
        expires_at: Optional[float] = None,
    ) -> List[ChunkRecord]:
        now = time.time()
        records: List[ChunkRecord] = []
        with self._lock, self._conn() as conn:
            for text in texts:
                cid = uuid.uuid4().hex
                h = content_hash(text)
                # Skip duplicate chunk text for same user
                dup = conn.execute(
                    "SELECT id FROM chunks WHERE user_id = ? AND content_hash = ?",
                    (user_id, h),
                ).fetchone()
                if dup:
                    continue
                conn.execute(
                    """
                    INSERT INTO chunks (id, source_id, source, text, kind, content_hash,
                                        user_id, created_at, expires_at, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cid, source_id, source, text, kind, h, user_id, now, expires_at, "{}"),
                )
                records.append(
                    ChunkRecord(
                        id=cid,
                        source_id=source_id,
                        source=source,
                        text=text,
                        kind=kind,
                        content_hash=h,
                        user_id=user_id,
                        created_at=now,
                        expires_at=expires_at,
                    )
                )
            conn.execute(
                "UPDATE sources SET chunk_count = (SELECT COUNT(*) FROM chunks WHERE source_id = ?), updated_at = ? WHERE id = ?",
                (source_id, now, source_id),
            )
        return records

    def list_sources(self, user_id: str = "default") -> List[SourceRecord]:
        self.purge_expired(user_id)
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def get_source(self, source_id: str, user_id: str = "default") -> Optional[SourceRecord]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ? AND user_id = ?",
                (source_id, user_id),
            ).fetchone()
        return self._row_to_source(row) if row else None

    def delete_source(self, source_id: str, user_id: str = "default") -> int:
        """Delete a source and its chunks. Returns number of chunks removed."""
        with self._lock, self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE source_id = ? AND user_id = ?",
                (source_id, user_id),
            ).fetchone()["c"]
            conn.execute(
                "DELETE FROM sources WHERE id = ? AND user_id = ?",
                (source_id, user_id),
            )
        return int(count)

    def clear(self, user_id: Optional[str] = None) -> None:
        with self._lock, self._conn() as conn:
            if user_id:
                conn.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM sources WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
            else:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM sources")
                conn.execute("DELETE FROM chat_messages")

    def load_chunks(self, user_id: str = "default") -> List[ChunkRecord]:
        self.purge_expired(user_id)
        now = time.time()
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE user_id = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at ASC
                """,
                (user_id, now),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def chunk_count(self, user_id: str = "default") -> int:
        with self._lock, self._conn() as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM chunks WHERE user_id = ?",
                    (user_id,),
                ).fetchone()["c"]
            )

    def all_chunk_ids(self, user_id: str = "default") -> List[str]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return [r["id"] for r in rows]

    # --- chat history ---

    def get_history(self, session_id: str, user_id: str = "default", limit: int = 12) -> List[Dict[str, str]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = ? AND user_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (session_id, user_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def append_history(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "default",
        max_turns: int = 12,
    ) -> None:
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, user_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, user_id, role, content, now),
            )
            # Trim old messages (keep last max_turns*2 rows roughly)
            keep = max_turns * 2
            conn.execute(
                """
                DELETE FROM chat_messages WHERE id IN (
                    SELECT id FROM chat_messages
                    WHERE session_id = ? AND user_id = ?
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (session_id, user_id, keep),
            )
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE id = ? AND user_id = ?",
                    (now, session_id, user_id),
                )
            else:
                default_title = content.strip()[:60] if role == "user" else "گفتگوی جدید"
                conn.execute(
                    """
                    INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, user_id, default_title, now, now),
                )

    def list_sessions(self, user_id: str = "default") -> List[Dict[str, Any]]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chat_sessions "
                "WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            out = []
            for r in rows:
                last = conn.execute(
                    "SELECT content FROM chat_messages WHERE session_id = ? AND user_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (r["id"], user_id),
                ).fetchone()
                out.append(
                    {
                        "id": r["id"],
                        "title": r["title"] or "گفتگوی جدید",
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                        "preview": (last["content"][:80] if last else ""),
                    }
                )
            return out

    def rename_session(self, session_id: str, title: str, user_id: str = "default") -> bool:
        title = (title or "").strip()[:80]
        if not title:
            return False
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ? AND user_id = ?",
                (title, session_id, user_id),
            )
            return cur.rowcount > 0

    def delete_session(self, session_id: str, user_id: str = "default") -> bool:
        with self._lock, self._conn() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            cur = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            return cur.rowcount > 0

    def clear_history(self, session_id: Optional[str] = None, user_id: str = "default") -> None:
        with self._lock, self._conn() as conn:
            if session_id:
                conn.execute(
                    "DELETE FROM chat_messages WHERE session_id = ? AND user_id = ?",
                    (session_id, user_id),
                )
                conn.execute(
                    "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                    (session_id, user_id),
                )
            else:
                conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))

    # --- crawl cache for re-crawl diffing ---

    def get_crawl_hash(self, url: str, user_id: str = "default") -> Optional[str]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT content_hash FROM crawl_pages WHERE url = ? AND user_id = ?",
                (url, user_id),
            ).fetchone()
        return row["content_hash"] if row else None

    def set_crawl_page(
        self, url: str, content_hash_val: str, title: str = "", user_id: str = "default"
    ) -> None:
        now = time.time()
        with self._lock, self._conn() as conn:
            conn.execute(
                """
                INSERT INTO crawl_pages (url, content_hash, title, last_crawled, user_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    title = excluded.title,
                    last_crawled = excluded.last_crawled,
                    user_id = excluded.user_id
                """,
                (url, content_hash_val, title, now, user_id),
            )

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> SourceRecord:
        meta = {}
        try:
            meta = json.loads(row["meta"] or "{}")
        except json.JSONDecodeError:
            pass
        return SourceRecord(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            content_hash=row["content_hash"],
            user_id=row["user_id"],
            chunk_count=int(row["chunk_count"] or 0),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            expires_at=row["expires_at"],
            meta=meta,
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> ChunkRecord:
        meta = {}
        try:
            meta = json.loads(row["meta"] or "{}")
        except json.JSONDecodeError:
            pass
        return ChunkRecord(
            id=row["id"],
            source_id=row["source_id"],
            source=row["source"],
            text=row["text"],
            kind=row["kind"],
            content_hash=row["content_hash"],
            user_id=row["user_id"],
            created_at=float(row["created_at"]),
            expires_at=row["expires_at"],
            meta=meta,
        )


_store: Optional[SQLiteStore] = None


def get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore()
    return _store

"""Central configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("JAVAD_DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_PATH = Path(os.getenv("JAVAD_SQLITE_PATH", str(DATA_DIR / "javad.db")))
CHROMA_PATH = Path(os.getenv("JAVAD_CHROMA_PATH", str(DATA_DIR / "chroma")))
CHROMA_COLLECTION = os.getenv("JAVAD_CHROMA_COLLECTION", "javad_chunks")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("JAVAD_OLLAMA_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("JAVAD_OLLAMA_EMBED_MODEL", "nomic-embed-text")

DEFAULT_LANGUAGE = os.getenv("JAVAD_DEFAULT_LANGUAGE", "fa")
API_KEY = os.getenv("JAVAD_API_KEY", "").strip()
AUTH_REQUIRED = os.getenv("JAVAD_AUTH_REQUIRED", "false").lower() in {"1", "true", "yes"}

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("JAVAD_CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
    if o.strip()
]
# Development convenience: empty or "*" means allow all
if os.getenv("JAVAD_CORS_ALLOW_ALL", "").lower() in {"1", "true", "yes"}:
    CORS_ORIGINS = ["*"]

CHUNK_MAX_LEN = int(os.getenv("JAVAD_CHUNK_MAX_LEN", "600"))
CHUNK_OVERLAP = int(os.getenv("JAVAD_CHUNK_OVERLAP", "80"))
RETRIEVAL_TOP_K = int(os.getenv("JAVAD_RETRIEVAL_TOP_K", "5"))
RETRIEVAL_CANDIDATES = int(os.getenv("JAVAD_RETRIEVAL_CANDIDATES", "20"))
RELEVANCE_THRESHOLD = float(os.getenv("JAVAD_RELEVANCE_THRESHOLD", "0.18"))
KNOWLEDGE_TTL_DAYS = int(os.getenv("JAVAD_KNOWLEDGE_TTL_DAYS", "0"))  # 0 = no expiry

MAX_UPLOAD_BYTES = int(os.getenv("JAVAD_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
RATE_LIMIT_UPLOAD = os.getenv("JAVAD_RATE_LIMIT_UPLOAD", "10/minute")
RATE_LIMIT_SCRAPE = os.getenv("JAVAD_RATE_LIMIT_SCRAPE", "5/minute")
RATE_LIMIT_CHAT = os.getenv("JAVAD_RATE_LIMIT_CHAT", "60/minute")

WHISPER_MODEL = os.getenv("JAVAD_WHISPER_MODEL", "base")
PIPER_MODEL_PATH = os.getenv("JAVAD_PIPER_MODEL", "")
VOICE_ENABLED = os.getenv("JAVAD_VOICE_ENABLED", "true").lower() in {"1", "true", "yes"}

# JAVAD — Roadmap to the Full-Feature Version

Current state: persistent RAG (SQLite + ChromaDB, hybrid BM25 + vector retrieval),
file/website ingestion with OCR/Office/audio, Ollama chat with streaming and offline
fallback, multi-language prompts (Persian default), optional API-key auth, real voice
channel (faster-whisper + Piper + VAD), docker-compose deployment.

---

## 1. Persistent storage (biggest gap — everything resets on restart)
- [x] Replace the in-memory `chunks: List[Chunk]` in [agent.py](backend/ollama_rag/agent.py) with a real datastore.
- [x] Add **SQLite** (simplest, zero extra services) or **Postgres** for chunk metadata (source, kind, timestamps, session).
- [x] Add a **vector store** for embeddings: start with `sqlite-vss` / `chromadb` (local, no server) or `qdrant` (if you want a service). Drop the hashed `_hash_sim` bag-of-words trick.
- [x] Persist chat history per `session_id` to the same DB instead of the in-memory `histories` dict.
- [x] On startup, load existing chunks/embeddings back into memory/cache for fast retrieval.

## 2. Real embeddings instead of hashed vectors
- [x] Call Ollama's embedding endpoint (e.g. `nomic-embed-text` or `mxbai-embed-large` pulled locally) in [client.py](backend/ollama_rag/client.py).
- [x] Replace `_hash_vec`/`_hash_sim` in [agent.py](backend/ollama_rag/agent.py) with real embedding vectors + cosine similarity (or let the vector store do it).
- [x] Re-embed chunks on ingest; store vectors alongside text in the vector store from step 1.
- [x] Add a nightly/on-demand re-index job for when the embedding model changes.

## 3. Better retrieval (RAG quality)
- [x] Add **hybrid search**: combine vector similarity with keyword/BM25 (e.g. `rank_bm25`) and merge-rank results.
- [x] Add **chunk overlap** in `_split_chunks` so facts split across chunk boundaries aren't lost.
- [x] Add **re-ranking** of top-k candidates (cross-encoder or a second Ollama call) before building context.
- [x] Track and show **source citations** in the chat response (already partially there via `sources`; surface them in the UI).
- [x] Add a **relevance threshold** so very low-scoring chunks are dropped instead of always returning top-5.

## 4. Richer document ingestion
- [x] Add **OCR** for images (`pytesseract` or `easyocr`) instead of the current metadata-only stub in [ingest.py:87](backend/rag/ingest.py:87).
- [x] Add `.docx`/`.pptx`/`.xlsx` extraction (`python-docx`, `python-pptx`, `openpyxl`).
- [x] Add `.csv`/`.json` structured ingestion with per-row/per-record chunking.
- [x] Add audio transcription ingestion (see Section 6) so voice notes/recordings become knowledge too.
- [x] Validate file size/type limits and reject unsupported/binary files cleanly (currently falls through to `"binary-as-text"`).

## 5. Smarter web crawling
- [x] Make `max_depth`/`max_pages` configurable per-request instead of hardcoded in [scraper.py:21](backend/ollama_rag/scraper.py:21).
- [x] Add `robots.txt` compliance before crawling.
- [x] Add scheduled **re-crawl** (via a cron task) to keep learned websites fresh, with diffing to avoid re-ingesting unchanged pages.
- [x] Support sitemaps (`sitemap.xml`) as a faster alternative to link-following.

## 6. Voice (turn the `/ws/voip-stream` stub into a real feature)
- [x] Integrate a local STT model (e.g. `faster-whisper`) to transcribe audio chunks received over the websocket in [main.py:469](backend/main.py:469).
- [x] Feed transcribed text into the existing `agent.chat()` pipeline.
- [x] Add local TTS (e.g. Piper) to speak answers back for a full voice loop.
- [x] Add VAD (voice activity detection) to segment streamed audio into utterances.

## 7. Multi-language support
- [x] Generalize the hardcoded Persian `SYSTEM_PROMPT` and leak-marker/validation heuristics in [agent.py:38](backend/ollama_rag/agent.py:38) behind a language setting.
- [x] Detect input language (e.g. `langdetect`) and switch system prompt/response language accordingly, or let the user pick a language in the UI.
- [x] Make `_is_chitchat` cues and `_LEAK_MARKERS` lists per-language instead of Persian-only.

## 8. API hardening & multi-user support
- [x] Add authentication (API key or session-based login) — currently anyone on the network can call every endpoint.
- [x] Scope knowledge/chat history per authenticated user instead of a single shared global agent instance (`get_agent()` singleton).
- [x] Add rate limiting on `/api/rag/upload` and `/api/website/scrape` to prevent abuse/DoS via large crawls.
- [x] Replace `allow_origins=["*"]` CORS in [main.py:44](backend/main.py:44) with an explicit allowlist for production.
- [x] Add structured logging + request IDs for debugging and auditing ingestion/chat events.

## 9. Memory management features
- [x] Add per-source deletion (`DELETE /api/rag/source/{id}`) instead of only full `memory_clear()`.
- [x] Add a "list knowledge" endpoint so the UI can show what's been learned (sources, chunk counts, dates).
- [x] Add knowledge expiry/TTL for stale web content.
- [x] Add dedup on ingest (hash-check content before adding) to avoid bloating the store with repeated uploads.

## 10. Frontend improvements
- [x] Add a knowledge browser UI (list/search/delete sources) backed by step 9's endpoints.
- [x] Add streaming chat responses (token-by-token) instead of waiting for the full Ollama reply.
- [x] Add citation display (clickable sources under each answer).
- [x] Add language/model switcher in the UI tied to step 7 and `JAVAD_OLLAMA_MODEL`.

## 11. Testing & observability
- [x] Add unit tests for `agent.ingest`/`agent.chat`/retrieval scoring.
- [x] Add integration tests for `/api/rag/upload`, `/api/website/scrape`, `/api/ollama/chat`.
- [x] Add a `/api/health` extension reporting vector store status, embedding model status, and chunk counts (partially present already in [main.py:506](backend/main.py:506)).
- [x] Add basic metrics (ingest latency, chat latency, fallback-mode rate) for self-monitoring.

## 12. Deployment
- [x] Containerize backend + Ollama with `docker-compose` for one-command startup.
- [x] Add a `.env.example` documenting `JAVAD_OLLAMA_MODEL` and any new config (DB path, embedding model, auth secret).
- [x] Document a production run profile (gunicorn/uvicorn workers, reverse proxy, TLS).

---

### Suggested order of attack
1. Persistence (1) → unlocks everything else surviving restarts.
2. Real embeddings (2) + better retrieval (3) → biggest answer-quality jump.
3. Richer ingestion (4) + smarter crawling (5) → more useful knowledge base.
4. API hardening (8) + memory management (9) → makes it safe to expose beyond localhost.
5. Voice (6), multi-language (7), frontend polish (10) → feature completeness.
6. Testing (11) and deployment (12) throughout, not saved for last.

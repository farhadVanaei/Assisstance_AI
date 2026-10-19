You are implementing the full feature roadmap for the JAVAD project (a local, offline,
Persian-language assistant built with FastAPI + Ollama + a RAG pipeline). The roadmap is
in `ROADMAP.md` at the repo root — read it in full before starting.

Context on the current codebase:
- `backend/main.py` — FastAPI app, all HTTP/WebSocket endpoints.
- `backend/ollama_rag/agent.py` — in-memory chunk store, retrieval, chat orchestration.
- `backend/ollama_rag/client.py` — Ollama HTTP client.
- `backend/ollama_rag/scraper.py` — domain-limited recursive web crawler.
- `backend/rag/ingest.py` — file text extraction (PDF, HTML, images, text).
- `frontend/` — plain HTML/CSS/JS UI, Persian-only.
- No database, no auth, no persistence — everything lives in process memory and is lost
  on restart. The `/ws/voip-stream` endpoint is a stub with no real STT/TTS.

Work through `ROADMAP.md` top to bottom, following its "Suggested order of attack"
section at the bottom of the file:
1. Persistence (SQLite/Postgres + a local vector store — chromadb or sqlite-vss)
2. Real embeddings via Ollama's embedding models (replace the hashed bag-of-words
   similarity in `agent.py`)
3. Retrieval quality (hybrid BM25 + vector search, chunk overlap, re-ranking, relevance
   threshold, surfaced citations)
4. Richer ingestion (OCR, docx/pptx/xlsx/csv) and smarter crawling (configurable
   depth/pages, robots.txt, sitemap support)
5. API hardening (auth, per-user scoping, rate limiting, explicit CORS allowlist,
   structured logging) and knowledge management (list/delete sources, dedup, TTL)
6. Voice (real STT via faster-whisper and TTS via Piper on the existing websocket),
   multi-language support (generalize the Persian-only prompts/heuristics), frontend
   polish (knowledge browser, streaming responses, citations UI, language/model
   switcher)
7. Tests (unit + integration) and deployment (docker-compose, `.env.example`, prod run
   profile) — add these alongside each phase above, not deferred to the end.

Requirements while implementing:
- Treat every unchecked box in `ROADMAP.md` as a task; check it off (`- [x]`) in that
  file as you complete it, in the same commit/PR as the corresponding code.
- Preserve existing behavior and the Persian UX by default — language generalization
  must not make Persian a second-class citizen; it should remain the default.
- Keep changes incremental and reviewable: prefer several focused commits over one giant
  diff, matching the roadmap's section boundaries.
- Add or update tests for every new piece of ingestion/retrieval/chat logic; do not mark
  a roadmap item done without a passing test or a manual verification note.
- Update `requirements.txt` and `README.md` whenever you add a dependency or change how
  the app is run (new env vars, new services like a vector DB, new setup steps).
- Do not introduce breaking changes to existing API routes/response shapes unless the
  roadmap explicitly calls for it (e.g. new auth) — add new endpoints alongside old ones
  where reasonable.
- Ask before making destructive changes (dropping data formats, removing endpoints) if
  the roadmap doesn't explicitly specify them.

When finished, summarize what was implemented, what was deferred and why, and the exact
steps to run the upgraded system locally (new env vars, new services to start, migration
steps for existing uploaded data if any).

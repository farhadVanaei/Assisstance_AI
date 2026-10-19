# دستیار عامل جواد (اولاما · آفلاین)

گفتگوی فارسی کاملاً محلی با **اولاما** + پایگاه دانش پایدار (SQLite + ChromaDB) + خزشگر محدود به دامنه + صوت اختیاری.

زبان پیش‌فرض: **فارسی** (با پشتیبانی اختیاری انگلیسی و عربی).

## ساختار

```
JAVAD/
├── backend/
│   ├── main.py
│   ├── data/                      # SQLite + Chroma (پایدار)
│   ├── uploads/
│   ├── rag/ingest.py              # استخراج متن (PDF/Office/OCR/صوت/…)
│   └── ollama_rag/
│       ├── agent.py               # بازیابی ترکیبی + گفتگو
│       ├── client.py              # اولاما (چت + امبدینگ + استریم)
│       ├── store.py               # SQLite (منابع، قطعات، تاریخچه)
│       ├── vector_store.py        # ChromaDB
│       ├── scraper.py             # خزش + robots + sitemap + بازخزش
│       ├── auth.py / config.py / languages.py / voice.py / …
├── frontend/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── ROADMAP.md
```

## رابط‌ها

| روش | مسیر | توضیح |
|-----|------|--------|
| `POST` | `/api/rag/upload` | فایل و/یا آدرس وب → دانش محلی |
| `POST` | `/api/rag/upload/stream` | همان، با جریان پیشرفت (NDJSON) |
| `POST` | `/api/website/scrape` | خزش دامنه (عمق/صفحات/sitemap/robots قابل تنظیم) |
| `POST` | `/api/ollama/chat` | گفتگو + زمینه + citations |
| `POST` | `/api/ollama/chat/stream` | پاسخ توکن‌به‌توکن (NDJSON) |
| `GET` | `/api/knowledge` | فهرست منابع دانش |
| `DELETE` | `/api/rag/source/{id}` | حذف یک منبع |
| `POST` | `/api/memory/clear` | پاکسازی کامل دانش کاربر |
| `POST` | `/api/reindex` | بازسازی امبدینگ‌ها |
| `GET` | `/api/health` | وضعیت اولاما، بردار، امبدینگ، متریک‌ها |
| `GET` | `/api/metrics` | تأخیر ingest/chat و نرخ fallback |
| `WS` | `/ws/voip-stream` | STT (faster-whisper) + گفتگو + TTS (Piper) |

احراز هویت اختیاری: هدر `X-API-Key` (و در صورت نیاز `X-User-Id`) وقتی `JAVAD_API_KEY` تنظیم شده باشد.

## اجرا (محلی)

### ۱) اولاما

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
```

### ۲) وابستگی‌ها و اجرا

```bash
cd /home/rome/PyCharmProjects/JAVAD
cp .env.example .env   # در صورت نیاز ویرایش کنید
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

باز کنید: **http://127.0.0.1:8000**

بدون اولاما، بارگذاری دانش کار می‌کند؛ گفتگو به حالت آفلاین (استخراج از قطعات) برمی‌گردد. بدون مدل امبدینگ، بردارهای hash-fallback استفاده می‌شوند.

### صوت (اختیاری)

```bash
pip install faster-whisper
# Piper: مدل .onnx را دانلود و مسیر را در JAVAD_PIPER_MODEL بگذارید
# OCR تصویر: tesseract (+ fas) یا easyocr
```

### تست‌ها

```bash
cd /home/rome/PyCharmProjects/JAVAD
source .venv/bin/activate
PYTHONPATH=backend pytest -q
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
# سپس مدل‌ها را بکشید:
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

داده‌ها در volume `javad_data` پایدار می‌مانند.

## متغیرهای محیطی مهم

نگاه کنید به [`.env.example`](.env.example). خلاصه:

| متغیر | پیش‌فرض | نقش |
|--------|---------|-----|
| `JAVAD_OLLAMA_MODEL` | `llama3.2` | مدل گفتگو |
| `JAVAD_OLLAMA_EMBED_MODEL` | `nomic-embed-text` | مدل امبدینگ |
| `JAVAD_SQLITE_PATH` / `JAVAD_CHROMA_PATH` | زیر `backend/data` | persistence |
| `JAVAD_API_KEY` | خالی | احراز هویت |
| `JAVAD_CORS_ORIGINS` | localhost:8000 | allowlist |
| `JAVAD_DEFAULT_LANGUAGE` | `fa` | زبان پیش‌فرض |
| `JAVAD_KNOWLEDGE_TTL_DAYS` | `0` | انقضای دانش وب (۰ = بدون TTL) |
| `JAVAD_PIPER_MODEL` | خالی | مسیر مدل TTS |

## مهاجرت از نسخهٔ قدیمی (حافظهٔ درحال‌اجرا)

نسخهٔ قبلی همهٔ دانش را در RAM نگه می‌داشت و با ری‌استارت از بین می‌رفت. در این نسخه دانش در SQLite + Chroma ذخیره می‌شود؛ **دادهٔ قبلی قابل بازیابی نیست** — اسناد/وب‌سایت‌ها را دوباره آپلود کنید. فایل‌های داخل `backend/uploads/` در صورت وجود دست‌نخورده می‌مانند ولی به‌طور خودکار دوباره ایندکس نمی‌شوند.

## پروفایل تولید

```bash
# چند worker پشت reverse proxy (مثلاً nginx با TLS)
cd backend
gunicorn main:app -k uvicorn.workers.UvicornWorker \
  -w 2 -b 0.0.0.0:8000 --timeout 180
```

پیشنهادها:

- `JAVAD_AUTH_REQUIRED=true` و یک `JAVAD_API_KEY` قوی تنظیم کنید.
- `JAVAD_CORS_ALLOW_ALL` را خاموش و `JAVAD_CORS_ORIGINS` را صریح بگذارید.
- TLS را روی nginx/caddy خاتمه دهید؛ به اپ فقط HTTP داخلی بدهید.
- Volume پایدار برای `/data` (SQLite + Chroma) و بکاپ منظم از `javad.db`.
- اولاما را روی همان میزبان یا شبکهٔ خصوصی نگه دارید؛ به اینترنت باز نکنید.

نصب gunicorn در صورت نیاز: `pip install gunicorn`.

## خزش وب

- فقط لینک‌های همان دامنه؛ احترام به `robots.txt`؛ کشف از `sitemap.xml` در صورت وجود.
- عمق و سقف صفحات از API قابل تنظیم است (پیش‌فرض: عمق ۲، ۲۰ صفحه).
- بازخزش زمان‌بندی‌شده با diff هش محتوا (بدون اینجست دوبارهٔ صفحات بدون تغییر).

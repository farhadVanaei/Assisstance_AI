"""Multi-language prompts and heuristics. Persian (fa) remains the default."""

from __future__ import annotations

from typing import Dict, List, Optional

from ollama_rag.config import DEFAULT_LANGUAGE

# System prompts per language
SYSTEM_PROMPTS: Dict[str, str] = {
    "fa": (
        "تو جواد هستی، یک دستیار هوشمند و محلی فارسی‌زبان. "
        "متن ارائه شده را بخوان و به پرسش کاربر به صورت طبیعی، کوتاه و به زبان فارسی روان پاسخ بده. "
        "از تکرار متن، نمایش لینک یا کدهای اضافی خودداری کن."
    ),
    "en": (
        "You are Javad, a smart local assistant. "
        "Read the provided text and answer the user's question naturally and briefly in English. "
        "Do not repeat the text, show raw links, or add extra code."
    ),
    "ar": (
        "أنت جواد، مساعد ذكي محلي. "
        "اقرأ النص المقدم وأجب على سؤال المستخدم بلغة عربية طبيعية ومختصرة. "
        "تجنب تكرار النص أو عرض الروابط أو الأكواد الإضافية."
    ),
}

EMPTY_PROMPTS: Dict[str, str] = {
    "fa": "لطفاً پیام خود را به فارسی بنویسید.",
    "en": "Please write your message.",
    "ar": "يرجى كتابة رسالتك.",
}

NO_KNOWLEDGE: Dict[str, str] = {
    "fa": (
        "هنوز چیزی در این باره در دانش من نیست. "
        "اگر سند یا وب‌سایتی دارید، اول در بخش پایگاه دانش اضافه کنید؛ "
        "بعد دوباره بپرسید."
    ),
    "en": (
        "I don't know about that yet. "
        "Please add a document or website in the knowledge section first, then ask again."
    ),
    "ar": (
        "لا أعرف عن ذلك بعد. "
        "يرجى إضافة مستند أو موقع في قسم المعرفة أولاً، ثم اسأل مرة أخرى."
    ),
}

FALLBACK_PREFIX: Dict[str, str] = {
    "fa": "طبق چیزی که یاد گرفته‌ام،",
    "en": "Based on what I've learned,",
    "ar": "بناءً على ما تعلمته،",
}

CHITCHAT_CUES: Dict[str, tuple] = {
    "fa": ("سلام", "درود", "خوبی", "حالت", "ممنون", "مرسی", "خداحافظ", "صبح بخیر", "شب خوش"),
    "en": ("hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good night", "how are you"),
    "ar": ("مرحبا", "السلام", "شكرا", "مع السلامة", "صباح الخير", "كيف حالك"),
}

LEAK_MARKERS: Dict[str, List[str]] = {
    "fa": [
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
    ],
    "en": [
        "knowledge context",
        "provided text",
        "user question",
        "do not repeat",
        "source:",
        "system:",
        "user:",
        "assistant:",
        "context:",
        "rag:",
    ],
    "ar": [
        "سياق المعرفة",
        "النص المقدم",
        "سؤال المستخدم",
        "system:",
        "user:",
        "assistant:",
        "context:",
    ],
}

SUPPORTED_LANGUAGES = ("fa", "en", "ar")


def normalize_lang(lang: Optional[str]) -> str:
    code = (lang or DEFAULT_LANGUAGE or "fa").strip().lower()[:2]
    return code if code in SUPPORTED_LANGUAGES else "fa"


def detect_language(text: str, fallback: Optional[str] = None) -> str:
    """Detect language from text; Persian remains default when unsure."""
    text = (text or "").strip()
    if not text:
        return normalize_lang(fallback)

    # Fast script heuristics (prefer over langdetect for short Persian)
    persian_chars = len([c for c in text if "\u0600" <= c <= "\u06FF"])
    arabic_specific = len([c for c in text if c in "ةىأإؤئ"])
    latin = len([c for c in text if ("a" <= c.lower() <= "z")])

    if persian_chars > 3 and arabic_specific < persian_chars * 0.15:
        return "fa"
    if arabic_specific > 2 and persian_chars > 3:
        return "ar"
    if latin > 8 and persian_chars == 0:
        return "en"

    try:
        from langdetect import detect

        code = detect(text)[:2]
        return normalize_lang(code if code in SUPPORTED_LANGUAGES else fallback)
    except Exception:
        return normalize_lang(fallback)


def system_prompt(lang: str) -> str:
    return SYSTEM_PROMPTS.get(normalize_lang(lang), SYSTEM_PROMPTS["fa"])


def empty_prompt(lang: str) -> str:
    return EMPTY_PROMPTS.get(normalize_lang(lang), EMPTY_PROMPTS["fa"])


def no_knowledge(lang: str) -> str:
    return NO_KNOWLEDGE.get(normalize_lang(lang), NO_KNOWLEDGE["fa"])


def fallback_prefix(lang: str) -> str:
    return FALLBACK_PREFIX.get(normalize_lang(lang), FALLBACK_PREFIX["fa"])


def chitchat_cues(lang: str) -> tuple:
    return CHITCHAT_CUES.get(normalize_lang(lang), CHITCHAT_CUES["fa"])


def leak_markers(lang: str) -> List[str]:
    # Always include shared English scaffolding markers
    base = list(LEAK_MARKERS.get(normalize_lang(lang), LEAK_MARKERS["fa"]))
    for m in LEAK_MARKERS["en"]:
        if m not in base:
            base.append(m)
    return base


def all_chitchat_cues() -> tuple:
    cues: List[str] = []
    for vals in CHITCHAT_CUES.values():
        cues.extend(vals)
    return tuple(cues)

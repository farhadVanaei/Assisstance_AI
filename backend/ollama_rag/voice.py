"""Local STT (faster-whisper) + TTS (Piper) + simple energy-based VAD."""

from __future__ import annotations

import asyncio
import io
import logging
import math
import struct
import wave
from pathlib import Path
from typing import Optional

from ollama_rag.config import PIPER_MODEL_PATH, VOICE_ENABLED, WHISPER_MODEL

logger = logging.getLogger("javad.voice")

_whisper_model = None
_whisper_lock = asyncio.Lock()


def _pcm16_rms(frame: bytes) -> int:
    """RMS of PCM16 LE mono samples (stdlib audioop was removed in Python 3.13)."""
    if len(frame) < 2:
        return 0
    n = len(frame) // 2
    samples = struct.unpack(f"<{n}h", frame[: n * 2])
    acc = 0
    for s in samples:
        acc += s * s
    return int(math.sqrt(acc / n))


def voice_status() -> dict:
    stt = False
    tts = False
    try:
        import faster_whisper  # noqa: F401

        stt = True
    except Exception:
        pass
    if PIPER_MODEL_PATH and Path(PIPER_MODEL_PATH).exists():
        tts = True
    else:
        try:
            import piper  # noqa: F401

            tts = True
        except Exception:
            pass
    return {"enabled": VOICE_ENABLED, "stt": stt, "tts": tts, "whisper_model": WHISPER_MODEL}


async def _get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    async with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        return _whisper_model


async def transcribe_audio_bytes(
    data: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = "fa",
) -> str:
    if not VOICE_ENABLED:
        return ""
    try:
        model = await _get_whisper()
    except Exception as exc:
        logger.warning("faster-whisper not available: %s", exc)
        return ""

    suffix = Path(filename).suffix.lower() or ".wav"
    tmp = Path("/tmp") / f"javad_stt_{id(data)}{suffix}"
    try:
        tmp.write_bytes(data)

        def _run() -> str:
            segments, _info = model.transcribe(
                str(tmp),
                language=language if language and language != "auto" else None,
                beam_size=1,
                vad_filter=True,
            )
            return " ".join(s.text.strip() for s in segments if s.text).strip()

        return await asyncio.to_thread(_run)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


async def synthesize_speech(text: str, language: str = "fa") -> Optional[bytes]:
    """Return WAV bytes via Piper if configured; else None."""
    if not VOICE_ENABLED or not (text or "").strip():
        return None
    model_path = PIPER_MODEL_PATH
    if not model_path or not Path(model_path).exists():
        logger.info("Piper model not configured (JAVAD_PIPER_MODEL)")
        return None

    def _run() -> bytes:
        try:
            from piper import PiperVoice

            voice = PiperVoice.load(model_path)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize(text, wf)
            return buf.getvalue()
        except Exception:
            # CLI fallback: piper --model … --output_file …
            import subprocess
            import tempfile

            out = Path(tempfile.mktemp(suffix=".wav"))
            try:
                subprocess.run(
                    ["piper", "--model", model_path, "--output_file", str(out)],
                    input=text.encode("utf-8"),
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                return out.read_bytes()
            finally:
                out.unlink(missing_ok=True)

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        logger.warning("TTS failed: %s", exc)
        return None


class EnergyVAD:
    """Simple RMS energy VAD to segment streamed PCM16 audio into utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        silence_ms: int = 600,
        threshold: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.threshold = threshold
        self._buf = bytearray()
        self._utterance = bytearray()
        self._silent = 0
        self._in_speech = False

    def feed(self, pcm16_bytes: bytes) -> list[bytes]:
        """Feed raw PCM16 LE mono bytes; return completed utterances (PCM)."""
        self._buf.extend(pcm16_bytes)
        done: list[bytes] = []
        while len(self._buf) >= self.frame_bytes:
            frame = bytes(self._buf[: self.frame_bytes])
            del self._buf[: self.frame_bytes]
            rms = _pcm16_rms(frame)
            if rms >= self.threshold:
                self._in_speech = True
                self._silent = 0
                self._utterance.extend(frame)
            elif self._in_speech:
                self._utterance.extend(frame)
                self._silent += 1
                if self._silent >= self.silence_frames:
                    if len(self._utterance) > self.frame_bytes * 3:
                        done.append(bytes(self._utterance))
                    self._utterance.clear()
                    self._in_speech = False
                    self._silent = 0
        return done

    def flush(self) -> Optional[bytes]:
        if self._in_speech and len(self._utterance) > self.frame_bytes * 3:
            data = bytes(self._utterance)
            self._utterance.clear()
            self._in_speech = False
            self._silent = 0
            return data
        self._utterance.clear()
        return None


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()

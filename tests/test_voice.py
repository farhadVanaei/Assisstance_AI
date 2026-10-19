"""Unit tests for energy VAD and PCM helpers (no whisper/piper required)."""

from __future__ import annotations

import struct

from ollama_rag.voice import EnergyVAD, _pcm16_rms, pcm16_to_wav


def test_pcm16_rms_silence():
    silence = b"\x00\x00" * 160
    assert _pcm16_rms(silence) == 0


def test_pcm16_rms_loud():
    samples = [10000] * 160
    frame = struct.pack(f"<{len(samples)}h", *samples)
    assert _pcm16_rms(frame) > 1000


def test_vad_segments_utterance():
    vad = EnergyVAD(sample_rate=16000, frame_ms=30, silence_ms=90, threshold=500)
    loud = struct.pack("<h", 8000) * (vad.frame_bytes // 2)
    quiet = b"\x00\x00" * (vad.frame_bytes // 2)

    # Speech for several frames
    for _ in range(5):
        assert vad.feed(loud) == []

    # Enough silence to close utterance
    done = []
    for _ in range(5):
        done.extend(vad.feed(quiet))
    assert done
    assert len(done[0]) > vad.frame_bytes * 3


def test_pcm16_to_wav_header():
    pcm = struct.pack("<h", 100) * 100
    wav = pcm16_to_wav(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF"
    assert b"WAVE" in wav[:16]

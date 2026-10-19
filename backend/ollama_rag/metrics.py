"""Simple in-process metrics for self-monitoring."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict


class Metrics:
    def __init__(self, window: int = 200) -> None:
        self._lock = threading.Lock()
        self.ingest_latencies: Deque[float] = deque(maxlen=window)
        self.chat_latencies: Deque[float] = deque(maxlen=window)
        self.ingest_count = 0
        self.chat_count = 0
        self.fallback_count = 0
        self.ollama_count = 0
        self.errors = 0

    def record_ingest(self, seconds: float) -> None:
        with self._lock:
            self.ingest_latencies.append(seconds)
            self.ingest_count += 1

    def record_chat(self, seconds: float, mode: str) -> None:
        with self._lock:
            self.chat_latencies.append(seconds)
            self.chat_count += 1
            if mode == "fallback":
                self.fallback_count += 1
            elif mode == "ollama":
                self.ollama_count += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            def _avg(vals: Deque[float]) -> float:
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            fallback_rate = (
                round(self.fallback_count / self.chat_count, 4) if self.chat_count else 0.0
            )
            return {
                "ingest_count": self.ingest_count,
                "chat_count": self.chat_count,
                "fallback_count": self.fallback_count,
                "ollama_count": self.ollama_count,
                "fallback_rate": fallback_rate,
                "avg_ingest_latency_s": _avg(self.ingest_latencies),
                "avg_chat_latency_s": _avg(self.chat_latencies),
                "errors": self.errors,
            }


metrics = Metrics()


class timed:
    """Context manager that returns elapsed seconds."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start

"""Egyszerű in-memory metrika-gyűjtő a /metrics endpoint-hoz.

A 4. napon érkező drift-detection alapja: a memóriában tartott legutóbbi
N predikció statisztikáit aggregáljuk és tesszük elérhetővé HTTP-n
keresztül. Pythonon belül, lock-mentesen működik (a Pythonnak van GIL,
és a műveletek atomiak listák/dict-ek esetén); ha többszálú lenne,
asyncio.Lock kellene.

Mit tárolunk?
- A legutóbbi N predikció minimális rekordját (label, confidence, latency,
  truncation, token-count, timestamp). Nem tároljuk az input szöveget —
  PII / GDPR szempont.
- A teljes futási idő óta kumulált összesítések (request_count, error_count).
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PredictionRecord:
    """Egy predikció minimális rekordja (input szöveg nélkül)."""

    timestamp: datetime
    predicted_label: str
    confidence: float
    latency_ms: float
    truncated: bool
    input_token_count: int
    request_id: Optional[str] = None


class MetricsCollector:
    """Thread-safe, fix kapacitású gyűjtő a recent predikciókhoz."""

    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._records: deque[PredictionRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._started_at: datetime = datetime.now(timezone.utc)

    def record_prediction(self, rec: PredictionRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._total_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def snapshot(self) -> dict:
        """Aggregált statisztikák a /metrics endpoint-hoz."""
        with self._lock:
            records = list(self._records)
            total_requests = self._total_requests
            total_errors = self._total_errors
            started_at = self._started_at

        if not records:
            return {
                "uptime_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "recent_window_size": 0,
                "predictions": None,
            }

        confidences = [r.confidence for r in records]
        latencies = [r.latency_ms for r in records]
        token_counts = [r.input_token_count for r in records]
        labels = [r.predicted_label for r in records]
        truncated_count = sum(1 for r in records if r.truncated)

        # Predikciós eloszlás (százalékban)
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        label_distribution = {
            label: round(count / total, 4)
            for label, count in label_counts.most_common()
        }

        return {
            "uptime_seconds": round(
                (datetime.now(timezone.utc) - started_at).total_seconds(), 1
            ),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "recent_window_size": len(records),
            "predictions": {
                "label_distribution": label_distribution,
                "confidence": {
                    "mean": round(sum(confidences) / len(confidences), 4),
                    "min": round(min(confidences), 4),
                    "max": round(max(confidences), 4),
                    "low_count": sum(1 for c in confidences if c < 0.5),
                    "low_ratio": round(
                        sum(1 for c in confidences if c < 0.5) / len(confidences), 4
                    ),
                },
                "latency_ms": {
                    "mean": round(sum(latencies) / len(latencies), 1),
                    "p50": round(_percentile(latencies, 50), 1),
                    "p95": round(_percentile(latencies, 95), 1),
                    "p99": round(_percentile(latencies, 99), 1),
                },
                "input_tokens": {
                    "mean": round(sum(token_counts) / len(token_counts), 1),
                    "max": max(token_counts),
                },
                "truncation": {
                    "count": truncated_count,
                    "ratio": round(truncated_count / len(records), 4),
                },
            },
        }


def _percentile(values: list[float], pct: float) -> float:
    """Egyszerű percentilis-számítás (numpy-mentes, hogy a metrika modul könnyű maradjon)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

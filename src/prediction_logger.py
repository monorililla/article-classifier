"""Strukturált predikciós logger.

Minden /classify hívás után egy JSON-soros rekord kerül a `logs/predictions.jsonl`
fájlba. Egy fájl-sor egy rekord (JSON Lines formátum), így streamelve is
feldolgozható (jq, pandas.read_json(lines=True), Loki, CloudWatch).

PII / GDPR szempont: az input szöveg NEM kerül a logba. Csak a SHA-256
hash-ének első 16 karaktere — visszakereséshez elég, de a tartalom nem
rekonstruálható.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.schemas import ClassificationResult


def _hash_text(text: str) -> str:
    """SHA-256 első 16 karaktere — input azonosításhoz, tartalom nélkül."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class PredictionLogger:
    """Append-only JSON Lines logger.

    A fájl egyszer lesz megnyitva append módban; minden record egy `\\n`-nel
    végződő JSON dokumentum. A flush minden record után kötelező — lekérdező
    eszközök (pl. tail -f) így azonnal látják az új sort.
    """

    def __init__(self, log_path: Path | str = "logs/predictions.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")
        self._py_logger = logging.getLogger(__name__)

    def log_prediction(
        self,
        result: ClassificationResult,
        input_text: str,
        client_id: Optional[str] = None,
    ) -> None:
        """Egy predikció rekordjának megírása."""
        record = {
            "timestamp": result.timestamp.isoformat(),
            "request_id": result.request_id,
            "client_id": client_id,
            "input": {
                "char_count": len(input_text),
                "token_count": result.input_token_count,
                "hash16": _hash_text(input_text),
                "truncated": result.truncated,
                "processed_token_count": result.processed_token_count,
            },
            "prediction": {
                "label": result.predicted_label,
                "confidence": round(result.confidence, 4),
                "all_scores": {k: round(v, 4) for k, v in result.all_scores.items()},
            },
            "performance": {
                "latency_ms": round(result.latency_ms, 1),
            },
            "version": {
                "model": result.model_version,
                "labels": result.labels_version,
                "code": result.code_version,
            },
        }
        self._write(record, level="INFO")

    def log_validation_error(
        self,
        request_id: str,
        error_type: str,
        message: str,
        input_text: Optional[str] = None,
    ) -> None:
        """Adatminőségi vagy validációs hiba rekordja."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "level": "WARNING",
            "event": "validation_error",
            "error_type": error_type,
            "message": message,
        }
        if input_text is not None:
            record["input"] = {
                "char_count": len(input_text),
                "hash16": _hash_text(input_text),
            }
        self._write(record, level="WARNING")

    def log_data_quality_warning(
        self,
        request_id: str,
        issue: str,
        details: Optional[dict] = None,
    ) -> None:
        """Nem-fatális adatminőségi figyelmeztetés (pl. túl rövid input,
        nem várt nyelv) — a request még feldolgozásra kerül, de jelezzük."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "level": "WARNING",
            "event": "data_quality_warning",
            "issue": issue,
        }
        if details:
            record["details"] = details
        self._write(record, level="WARNING")

    def _write(self, record: dict, level: str = "INFO") -> None:
        """Egyetlen JSON-sor írása a fájlba."""
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:
            self._py_logger.exception("Failed to write prediction log record")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # pragma: no cover
            pass

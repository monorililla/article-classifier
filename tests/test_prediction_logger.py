"""PredictionLogger tesztek.

A logger temp-fájlba ír, így izoláltan tesztelhető. A tesztek a JSON Lines
formátum helyességét és az érzékeny mezők (input szöveg) hiányát ellenőrzik.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.prediction_logger import PredictionLogger, _hash_text
from src.schemas import ClassificationResult


def make_result(text_for_token_count: str = "x" * 30) -> ClassificationResult:
    """Faked ClassificationResult egy logger-teszthez."""
    return ClassificationResult(
        predicted_label="sports",
        confidence=0.85,
        all_scores={"sports": 0.85, "business": 0.10, "politics": 0.05},
        truncated=False,
        input_token_count=len(text_for_token_count.split()),
        processed_token_count=len(text_for_token_count.split()),
        model_version="facebook/bart-large-mnli",
        labels_version="v1",
        code_version="0.2.0",
        latency_ms=234.5,
        timestamp=datetime.now(timezone.utc),
        request_id="test-req-001",
    )


def test_log_prediction_writes_jsonl_line(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    text = "The Lakers defeated the Celtics tonight."
    logger.log_prediction(make_result(text), input_text=text)
    logger.close()

    # A fájl egy soros JSON-t kell tartalmazzon
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["request_id"] == "test-req-001"
    assert record["prediction"]["label"] == "sports"
    assert record["prediction"]["confidence"] == 0.85
    assert record["version"]["labels"] == "v1"
    assert record["performance"]["latency_ms"] == 234.5


def test_log_prediction_does_not_leak_input_text(tmp_path: Path) -> None:
    """A logban NE legyen benne az input szöveg, csak a hash."""
    log_path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    secret_text = "This is the secret article that must not appear in logs."
    logger.log_prediction(make_result(secret_text), input_text=secret_text)
    logger.close()

    content = log_path.read_text()
    assert "secret article" not in content
    assert "must not appear" not in content

    # De a hash-nek benne kell lennie
    record = json.loads(content.strip())
    assert record["input"]["hash16"] == _hash_text(secret_text)


def test_log_validation_error_writes_warning(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    logger.log_validation_error(
        request_id="bad-req-1",
        error_type="ValueError",
        message="A `text` nem lehet üres string.",
        input_text="",
    )
    logger.close()

    record = json.loads(log_path.read_text().strip())
    assert record["level"] == "WARNING"
    assert record["event"] == "validation_error"
    assert record["error_type"] == "ValueError"


def test_log_data_quality_warning(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    logger.log_data_quality_warning(
        request_id="req-2",
        issue="input_too_short",
        details={"char_count": 8, "min_recommended": 50},
    )
    logger.close()

    record = json.loads(log_path.read_text().strip())
    assert record["level"] == "WARNING"
    assert record["event"] == "data_quality_warning"
    assert record["issue"] == "input_too_short"
    assert record["details"]["char_count"] == 8


def test_multiple_records_are_appended_as_separate_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    logger = PredictionLogger(log_path)

    for i in range(5):
        text = f"Article number {i}"
        logger.log_prediction(make_result(text), input_text=text)
    logger.close()

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 5
    # Minden sor érvényes JSON
    for line in lines:
        json.loads(line)


def test_hash_is_deterministic_and_short() -> None:
    h1 = _hash_text("hello world")
    h2 = _hash_text("hello world")
    h3 = _hash_text("HELLO WORLD")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16

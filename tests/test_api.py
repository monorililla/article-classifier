"""API szintű tesztek.

Itt nem a valódi modellt használjuk: a `Depends(get_classifier)` függőséget
felülbíráljuk egy MockClassifier-rel. Így a tesztek néhány másodperc alatt
lefutnak, és a fókusz az API-réteg viselkedésén van (request/response,
status kódok, hibakezelés), nem a modellen.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_classifier
from src.monitoring import MetricsCollector
from src.schemas import ClassificationResult


class MockClassifier:
    """Determinisztikus, gyors helyettesítő ArticleClassifier helyett."""

    labels_version = "v1"
    labels = ["sports", "business", "politics"]

    @property
    def info(self) -> dict:
        return {
            "model_name": "mock-model",
            "labels_version": "v1",
            "labels": self.labels,
            "code_version": "test",
            "device": "cpu",
            "model_max_tokens": 1024,
            "truncation_token_budget": 900,
        }

    def classify(self, text: str, labels=None, request_id=None) -> ClassificationResult:
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("A `text` nem lehet üres string.")
        used_labels = labels if labels is not None else self.labels
        # Egyszerű mock: a "sports" mindig nyer
        scores = {label: 0.1 for label in used_labels}
        if "sports" in used_labels:
            scores["sports"] = 0.8
            top = "sports"
        else:
            top = used_labels[0]
            scores[top] = 0.8

        # A scores összegét 1.0-ra normalizáljuk
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}

        return ClassificationResult(
            predicted_label=top,
            confidence=scores[top],
            all_scores=scores,
            truncated=False,
            input_token_count=len(text.split()),
            processed_token_count=len(text.split()),
            model_version="mock-model",
            labels_version="v1",
            code_version="test",
            latency_ms=1.0,
            timestamp=datetime.now(timezone.utc),
            request_id=request_id,
        )


@pytest.fixture
def client(tmp_path) -> TestClient:
    """TestClient mock classifierrel — nincs modell-betöltés."""
    from src.prediction_logger import PredictionLogger

    mock = MockClassifier()
    metrics = MetricsCollector(capacity=100)

    # PredictionLogger egy temp fájlba ír — a teszt után elhasználódik
    pred_logger = PredictionLogger(log_path=tmp_path / "predictions.jsonl")

    # Az app.state-be tesszük (ahogy a lifespan tenné), és a Depends átveszi
    app.state.classifier = mock
    app.state.metrics = metrics
    app.state.prediction_logger = pred_logger

    # A get_classifier override is megmarad biztonsági okból
    app.dependency_overrides[get_classifier] = lambda: mock

    yield TestClient(app)

    app.dependency_overrides.clear()
    pred_logger.close()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["labels_version"] == "v1"


def test_version_endpoint(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "mock-model"
    assert body["labels_version"] == "v1"
    assert "sports" in body["labels"]


def test_classify_endpoint_returns_prediction(client: TestClient) -> None:
    response = client.post(
        "/classify",
        json={"text": "The Lakers defeated the Celtics 112-104."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "sports"
    assert 0.0 <= body["confidence"] <= 1.0
    assert "request_id" in body  # autogenerált


def test_classify_endpoint_with_custom_labels(client: TestClient) -> None:
    response = client.post(
        "/classify",
        json={
            "text": "test",
            "labels": ["business", "politics"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_label"] == "business"  # mock: első label nyer ha nincs sports


def test_classify_endpoint_rejects_empty_text(client: TestClient) -> None:
    """Pydantic min_length=1 → 422."""
    response = client.post("/classify", json={"text": ""})
    assert response.status_code == 422


def test_classify_endpoint_rejects_missing_text(client: TestClient) -> None:
    response = client.post("/classify", json={})
    assert response.status_code == 422


def test_metrics_starts_empty_then_records_predictions(client: TestClient) -> None:
    # Friss state-et akarunk, ezért újrainicializáljuk a metrics-et
    app.state.metrics = MetricsCollector(capacity=100)

    # Üres állapot
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] == 0
    assert body["recent_window_size"] == 0

    # 3 predikció
    for _ in range(3):
        client.post("/classify", json={"text": "Sports article."})

    # Most már van adat
    response = client.get("/metrics")
    body = response.json()
    assert body["total_requests"] == 3
    assert body["recent_window_size"] == 3
    assert body["predictions"]["label_distribution"]["sports"] == 1.0
    assert body["predictions"]["confidence"]["mean"] > 0.0


def test_classify_endpoint_passes_request_id(client: TestClient) -> None:
    custom_id = "test-request-123"
    response = client.post(
        "/classify",
        json={"text": "Test", "request_id": custom_id},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == custom_id

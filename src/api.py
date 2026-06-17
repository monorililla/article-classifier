"""FastAPI alkalmazás az ArticleClassifier köré.

A modellt alkalmazás-induláskor töltjük be (lifespan event handler),
nem requestenként — ez kritikus a latency-szempontból:
hidegen ~5 mp az inicializálás, melegen ~250 ms egy predikció.

Endpoint-ok:
- POST /classify  — egy cikk osztályozása
- GET  /health    — healthcheck (Docker / load balancer)
- GET  /version   — verzió-információk (modell, címke-set, kód)
- GET  /metrics   — aggregált statisztikák a recent predikciókról

Az API automatikusan generál Swagger UI-t a /docs útvonalon,
ReDoc-ot a /redoc-on.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.config import CODE_VERSION
from src.monitoring import MetricsCollector, PredictionRecord
from src.pipeline import ArticleClassifier
from src.prediction_logger import PredictionLogger
from src.schemas import (
    ClassifyRequest,
    ClassificationResult,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# --- Globális komponensek (lifespan során töltjük) ---
# Lifespan-szemlélet: csak az `app.state` jelent állapotot, a globálisokat
# ne használjuk közvetlenül a routes-ban. A dependency injection (Depends)
# garantálja, hogy a tesztelés is reprodukálható.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modell betöltése app induláskor, cleanup app leállásakor."""
    logger.info("Alkalmazás indul — ArticleClassifier inicializálása...")
    app.state.classifier = ArticleClassifier()
    app.state.metrics = MetricsCollector(capacity=1000)
    app.state.prediction_logger = PredictionLogger(
        log_path=Path("logs/predictions.jsonl")
    )
    logger.info("Inicializálás kész.")
    yield
    logger.info("Alkalmazás leáll.")
    app.state.prediction_logger.close()


app = FastAPI(
    title="Article Classifier API",
    description=(
        "Zero-shot cikk-kategorizáló a `facebook/bart-large-mnli` modell "
        "köré épített REST API."
    ),
    version=CODE_VERSION,
    lifespan=lifespan,
)


# --- Dependency providers ---

def get_classifier(request: Request) -> ArticleClassifier:
    return request.app.state.classifier


def get_metrics(request: Request) -> MetricsCollector:
    return request.app.state.metrics


def get_prediction_logger(request: Request) -> PredictionLogger:
    return request.app.state.prediction_logger


# --- Endpoint-ok ---

@app.get("/health", response_model=HealthResponse, tags=["status"])
def health(
    classifier: ArticleClassifier = Depends(get_classifier),
) -> HealthResponse:
    """Healthcheck. Akkor 'ok', ha a modell betöltődött."""
    return HealthResponse(
        status="ok" if classifier else "loading",
        model_loaded=classifier is not None,
        code_version=CODE_VERSION,
        labels_version=classifier.labels_version,
    )


@app.get("/version", response_model=VersionResponse, tags=["status"])
def version(
    classifier: ArticleClassifier = Depends(get_classifier),
) -> VersionResponse:
    """A futó komponensek verzió-információi."""
    info = classifier.info
    return VersionResponse(
        code_version=info["code_version"],
        model_name=info["model_name"],
        labels_version=info["labels_version"],
        labels=info["labels"],
        device=info["device"],
        model_max_tokens=info["model_max_tokens"],
        truncation_token_budget=info["truncation_token_budget"],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["status"])
def metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics),
) -> MetricsResponse:
    """Aggregált statisztikák a recent predikciókról.

    Drift-detection alapja: ha a confidence-eloszlás vagy a
    predikciós címke-eloszlás jelentősen eltér a baseline-tól
    (lásd notebook/evaluation.ipynb), külső monitorozó tool
    riasztást emelhet ezeken az értékeken.
    """
    snapshot = metrics_collector.snapshot()
    return MetricsResponse(**snapshot)


@app.post(
    "/classify",
    response_model=ClassificationResult,
    responses={
        422: {"model": ErrorResponse, "description": "Érvénytelen input"},
        500: {"model": ErrorResponse, "description": "Belső hiba"},
    },
    tags=["classification"],
)
def classify(
    payload: ClassifyRequest,
    classifier: ArticleClassifier = Depends(get_classifier),
    metrics_collector: MetricsCollector = Depends(get_metrics),
    prediction_logger: PredictionLogger = Depends(get_prediction_logger),
) -> ClassificationResult:
    """Egy cikk osztályozása zero-shot megközelítéssel.

    Ha a `labels` mező None, a labels_v1.json-ben definiált címke-listát
    használjuk. Ha az input túl hosszú (>900 token), a pipeline automatikusan
    csonkolja és a válaszban `truncated=true` jelzi.
    """
    request_id = payload.request_id or str(uuid.uuid4())

    try:
        result = classifier.classify(
            text=payload.text,
            labels=payload.labels,
            request_id=request_id,
        )
    except ValueError as e:
        metrics_collector.record_error()
        prediction_logger.log_validation_error(
            request_id=request_id,
            error_type="ValueError",
            message=str(e),
            input_text=payload.text,
        )
        logger.warning("Validation error (request_id=%s): %s", request_id, e)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # pragma: no cover — generic safety net
        metrics_collector.record_error()
        prediction_logger.log_validation_error(
            request_id=request_id,
            error_type=type(e).__name__,
            message=str(e),
        )
        logger.exception("Unexpected error (request_id=%s)", request_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Sikeres predikció rögzítése a metrikákba és a struktúrált logba
    metrics_collector.record_prediction(
        PredictionRecord(
            timestamp=result.timestamp,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            truncated=result.truncated,
            input_token_count=result.input_token_count,
            request_id=request_id,
        )
    )
    prediction_logger.log_prediction(result=result, input_text=payload.text)

    return result


# --- Egyedi exception handler — egységes hibaformátum ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc.detail),
        ).model_dump(),
    )

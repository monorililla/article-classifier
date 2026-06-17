"""Pydantic schemák a pipeline és API számára.

Egy közös helyen tartjuk a request/response/result modelleket, így
a pipeline-belső eredmények és az API-bemenet/-kimenet azonos
struktúrát követnek.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Egy cikk osztályozásának eredménye."""

    predicted_label: str = Field(..., description="A legmagasabb pontszámú címke.")
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="A predikció biztossága (0-1).")
    all_scores: dict[str, float] = Field(...,
                                          description="Összes címke és pontszám.")
    truncated: bool = Field(..., description="Truncation történt-e.")
    input_token_count: int = Field(..., description="Az eredeti input token-száma.")
    processed_token_count: int = Field(...,
                                        description="A modellbe ténylegesen ment token-szám.")
    model_version: str = Field(..., description="A modell verziója (HF model ID).")
    labels_version: str = Field(..., description="A labels_v*.json verziója.")
    code_version: str = Field(..., description="A kód verziója (config.CODE_VERSION).")
    latency_ms: float = Field(..., description="Az inference ideje millisecondban.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                 description="A predikció időpontja (UTC).")
    request_id: Optional[str] = Field(None, description="Opcionális request azonosító.")


class ClassifyRequest(BaseModel):
    """API: bejövő kérés egy szöveg osztályozására."""

    text: str = Field(..., min_length=1, max_length=50_000,
                      description="Az osztályozandó cikk szövege.")
    labels: Optional[list[str]] = Field(
        None,
        description="Opcionálisan felülbírált candidate labels lista. "
                     "Ha None, a labels_v1.json-ból betöltött lista használt.",
    )
    request_id: Optional[str] = Field(None, description="Opcionális request azonosító.")


class HealthResponse(BaseModel):
    """API: /health endpoint válasza."""

    status: str
    model_loaded: bool
    code_version: str
    labels_version: str


class VersionResponse(BaseModel):
    """API: /version endpoint válasza — a futó komponensek azonosítói."""

    code_version: str
    model_name: str
    labels_version: str
    labels: list[str]
    device: str
    model_max_tokens: int
    truncation_token_budget: int


class MetricsResponse(BaseModel):
    """API: /metrics endpoint válasza — szabad-formájú dict.

    Pydantic-os tipizálást szándékosan kerüljük (a struktúra
    bővülni fog, és a dict típus elég flexibilis).
    """

    uptime_seconds: float
    total_requests: int
    total_errors: int
    recent_window_size: int
    predictions: Optional[dict] = None


class ErrorResponse(BaseModel):
    """API: hibaüzenet egységes formában."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None

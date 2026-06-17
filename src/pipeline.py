"""Article Classifier — production-grade wrapper a HF zero-shot pipeline körül.

Ez a modul az alapja a notebook-os kiértékelésnek és a FastAPI
endpoint-nak is. Egy importálható osztályt ad: ArticleClassifier.

Felelősségi körök:
- A HF zero-shot pipeline betöltése (modell, tokenizer)
- Címke-konfiguráció betöltése labels_vN.json-ből
- Token-szintű explicit truncation (hosszú cikkekhez)
- Strukturált eredmény (ClassificationResult) visszaadása
- Verzió-információk (modell, címke-set, kód) beágyazása minden válaszba

Minimális használat:

    from src.pipeline import ArticleClassifier

    clf = ArticleClassifier()
    result = clf.classify("Apple unveiled its new M5 chip today...")
    print(result.predicted_label, result.confidence)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, pipeline

from src.config import (
    CODE_VERSION,
    DEFAULT_LABELS_FILE,
    DEFAULT_MODEL,
    MODEL_MAX_TOKENS,
    TRUNCATION_TOKEN_BUDGET,
)
from src.schemas import ClassificationResult

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Kiválasztja a leggyorsabb elérhető eszközt."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ArticleClassifier:
    """Zero-shot cikk-kategorizáló a `facebook/bart-large-mnli` modellel.

    Tipikus használat:
        clf = ArticleClassifier()
        result = clf.classify(text)

    Egyedi labels-fájl és modell:
        clf = ArticleClassifier(
            model_name="facebook/bart-large-mnli",
            labels_file="data/prompts/labels_v2.json",
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        labels_file: Optional[Path | str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.labels_file = Path(labels_file) if labels_file else DEFAULT_LABELS_FILE
        self.device = device or _select_device()

        # 1) Címke-konfiguráció betöltése
        logger.info("Címke-konfiguráció betöltése: %s", self.labels_file)
        with open(self.labels_file, "r", encoding="utf-8") as f:
            self._labels_config = json.load(f)
        self.labels: list[str] = list(self._labels_config["labels"])
        self.labels_version: str = self._labels_config.get("version", "unknown")
        self.hypothesis_template: str = self._labels_config.get(
            "hypothesis_template", "This text is about {}."
        )

        # 2) Tokenizer betöltése (a truncation-hez kell)
        logger.info("Tokenizer betöltése: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 3) Pipeline betöltése
        logger.info("Pipeline betöltése (eszköz: %s)...", self.device)
        self._pipeline = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )
        logger.info("ArticleClassifier kész — %d címke, modell=%s, kód=%s",
                    len(self.labels), self.model_name, CODE_VERSION)

    # ---- Public API ------------------------------------------------------

    def classify(
        self,
        text: str,
        labels: Optional[list[str]] = None,
        request_id: Optional[str] = None,
    ) -> ClassificationResult:
        """Egy szöveget osztályoz és strukturált eredményt ad vissza.

        Args:
            text: Az osztályozandó szöveg.
            labels: Opcionálisan felülbírálja a default címke-listát
                    (pl. teszteléshez vagy szűkebb scope-hoz).
            request_id: Opcionális azonosító, ha logolni vagy vissza-
                        követni akarod a hívást.

        Returns:
            ClassificationResult — minden lényeges mező feltöltve.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("A `text` nem lehet üres string.")

        used_labels = labels if labels is not None else self.labels
        if not used_labels:
            raise ValueError("A címke-lista nem lehet üres.")

        # Truncation
        original_token_count = self._token_count(text)
        truncated = original_token_count > TRUNCATION_TOKEN_BUDGET
        processed_text = (
            self._truncate(text) if truncated else text
        )
        processed_token_count = self._token_count(processed_text)

        # Inference (latency mérés)
        t0 = time.perf_counter()
        raw = self._pipeline(
            processed_text,
            candidate_labels=used_labels,
            hypothesis_template=self.hypothesis_template,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        all_scores = dict(zip(raw["labels"], raw["scores"]))

        return ClassificationResult(
            predicted_label=raw["labels"][0],
            confidence=raw["scores"][0],
            all_scores=all_scores,
            truncated=truncated,
            input_token_count=original_token_count,
            processed_token_count=processed_token_count,
            model_version=self.model_name,
            labels_version=self.labels_version,
            code_version=CODE_VERSION,
            latency_ms=latency_ms,
            request_id=request_id,
        )

    # ---- Belső segédmetódusok -------------------------------------------

    def _token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate(self, text: str) -> str:
        """Levágja a szöveget az első TRUNCATION_TOKEN_BUDGET tokenig.

        Egyszerű stratégia: az eleje marad. Hosszú cikkeknél a vége
        elveszik — a hírcikkeknél ez sokszor elfogadható, mert a
        legfontosabb információ az első bekezdésekben van (lead-paragraph
        konvenció). Komplexebb stratégia (chunking + averaging) lehetne
        egy v2 továbbfejlesztés.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= TRUNCATION_TOKEN_BUDGET:
            return text
        truncated_tokens = tokens[:TRUNCATION_TOKEN_BUDGET]
        return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    @property
    def info(self) -> dict:
        """Verzió-információk diagnosztikához és /version endpoint-hoz."""
        return {
            "model_name": self.model_name,
            "labels_version": self.labels_version,
            "labels": self.labels,
            "code_version": CODE_VERSION,
            "device": self.device,
            "model_max_tokens": MODEL_MAX_TOKENS,
            "truncation_token_budget": TRUNCATION_TOKEN_BUDGET,
        }

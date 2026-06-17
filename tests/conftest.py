"""Közös pytest fixture-ök.

A `classifier` fixture session-scope-ú, azaz a teljes teszt-futtatás során
egyszer töltjük be a modellt. Az inicializálás kb. 5 mp MPS-en, ezért
minden tesztet egy közös classifierre futtatunk.
"""

from __future__ import annotations

import pytest

from src.pipeline import ArticleClassifier


@pytest.fixture(scope="session")
def classifier() -> ArticleClassifier:
    """Egyszer betöltött ArticleClassifier a teljes teszt-szekcióhoz."""
    return ArticleClassifier()

"""Pipeline szintű tesztek.

A tesztek a tényleges HF modellt használják (mockolás itt nem érdemes).
A classifier session-scope-ú fixture, így csak egyszer töltjük be.
"""

from __future__ import annotations

import pytest

from src.pipeline import ArticleClassifier


def test_classifier_initializes(classifier: ArticleClassifier) -> None:
    """Az inicializálás kész: 6 címke betöltve a labels_v1.json-ból."""
    assert classifier.labels_version == "v1"
    assert len(classifier.labels) == 6
    assert "sports" in classifier.labels
    assert "health" in classifier.labels


def test_classify_short_article(classifier: ArticleClassifier) -> None:
    """Egy egyszerű sport-cikk: a predikció sports legyen, magas confidence-szel."""
    text = "The Lakers defeated the Celtics 112-104 in overtime, with LeBron James scoring 35 points."
    result = classifier.classify(text)

    assert result.predicted_label == "sports"
    assert result.confidence > 0.5
    assert not result.truncated
    assert result.input_token_count > 0
    assert result.processed_token_count == result.input_token_count
    assert result.latency_ms > 0


def test_classify_long_article_triggers_truncation(classifier: ArticleClassifier) -> None:
    """1024 token feletti input: truncated=true, processed token <= 900."""
    # Generálunk egy biztosan 1024+ tokenes szöveget
    short = "Apple unveiled its new M5 chip with 40 percent better performance. "
    text = short
    while classifier._token_count(text) <= 1024:
        text += short

    result = classifier.classify(text)

    assert result.truncated is True
    assert result.input_token_count > 1024
    assert result.processed_token_count <= 900


def test_classify_empty_string_raises(classifier: ArticleClassifier) -> None:
    """Üres string ValueError-t dob."""
    with pytest.raises(ValueError):
        classifier.classify("")


def test_classify_whitespace_only_raises(classifier: ArticleClassifier) -> None:
    """Csak whitespace szintén érvénytelen."""
    with pytest.raises(ValueError):
        classifier.classify("   \n\t  ")


def test_classify_with_custom_labels(classifier: ArticleClassifier) -> None:
    """Felhasználó által megadott labels listával is működik."""
    text = "The Senate passed a bipartisan infrastructure bill on Thursday."
    custom_labels = ["politics", "cooking", "music"]
    result = classifier.classify(text, labels=custom_labels)

    assert result.predicted_label in custom_labels
    assert set(result.all_scores.keys()) == set(custom_labels)


def test_classify_returns_all_scores_summing_to_one(classifier: ArticleClassifier) -> None:
    """A scores közelítőleg 1.0-ra összegződik (zero-shot pipeline normalizál)."""
    text = "Researchers developed a new battery technology that could double EV range."
    result = classifier.classify(text)

    total = sum(result.all_scores.values())
    assert abs(total - 1.0) < 0.01


def test_classify_returns_versioning_info(classifier: ArticleClassifier) -> None:
    """A válasz tartalmazza a verzió-mezőket (modell, címke, kód)."""
    result = classifier.classify("Just a short test article.")

    assert result.model_version == "facebook/bart-large-mnli"
    assert result.labels_version == "v1"
    assert result.code_version  # nem-üres

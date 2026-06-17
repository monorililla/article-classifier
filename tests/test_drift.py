"""Drift-detection tesztek."""

from __future__ import annotations

import math

import pytest

from src.drift import assess_drift, kl_divergence, per_label_share_drift


def make_baseline() -> dict:
    return {
        "version": "test",
        "label_distribution": {
            "sports": 0.20,
            "business": 0.46,
            "world news": 0.19,
            "science and technology": 0.04,
            "politics": 0.10,
            "health": 0.01,
        },
        "confidence": {"mean": 0.61, "min_acceptable_mean": 0.50},
        "latency_ms": {"p50_baseline": 220, "p95_baseline": 460,
                        "p95_alert_threshold": 700},
        "drift_thresholds": {
            "kl_divergence_warning": 0.15,
            "kl_divergence_alert": 0.30,
            "label_share_drift_warning": 0.10,
            "label_share_drift_alert": 0.20,
        },
    }


def make_snapshot(label_dist: dict, conf_mean: float = 0.62,
                   p95: float = 460.0, n: int = 100) -> dict:
    return {
        "uptime_seconds": 100.0,
        "total_requests": n,
        "total_errors": 0,
        "recent_window_size": n,
        "predictions": {
            "label_distribution": label_dist,
            "confidence": {"mean": conf_mean, "min": 0.3, "max": 0.95,
                            "low_count": 5, "low_ratio": 0.05},
            "latency_ms": {"mean": 250, "p50": 240, "p95": p95, "p99": 600},
            "input_tokens": {"mean": 60, "max": 132},
            "truncation": {"count": 0, "ratio": 0.0},
        },
    }


def test_kl_divergence_zero_for_identical_distributions() -> None:
    p = {"a": 0.5, "b": 0.5}
    assert kl_divergence(p, p) < 1e-5


def test_kl_divergence_positive_for_different_distributions() -> None:
    p = {"a": 0.9, "b": 0.1}
    q = {"a": 0.1, "b": 0.9}
    assert kl_divergence(p, q) > 0.5


def test_per_label_share_drift_zero_for_identical() -> None:
    p = {"sports": 0.5, "business": 0.5}
    drift = per_label_share_drift(p, p)
    assert all(v < 1e-5 for v in drift.values())


def test_assess_drift_status_ok_when_close_to_baseline() -> None:
    baseline = make_baseline()
    snapshot = make_snapshot(baseline["label_distribution"], conf_mean=0.62)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] == "ok"


def test_assess_drift_warning_for_moderate_label_shift() -> None:
    baseline = make_baseline()
    # Sport-eloszlás 0.20 → 0.35 (15%-pont eltérés, warning küszöb 10%-pont)
    shifted = dict(baseline["label_distribution"])
    shifted["sports"] = 0.35
    shifted["business"] = 0.31  # csökkentés: business volt 0.46
    snapshot = make_snapshot(shifted, conf_mean=0.60)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] in {"warning", "alert"}
    # Az indoklás említi az eltolt címkék közül legalább egyet
    reasons_text = " ".join(result["reasons"])
    assert "sports" in reasons_text or "business" in reasons_text


def test_assess_drift_alert_for_drastic_label_shift() -> None:
    baseline = make_baseline()
    # Mindenből sport-cikk
    shifted = {k: 0.0 for k in baseline["label_distribution"]}
    shifted["sports"] = 1.0
    snapshot = make_snapshot(shifted, conf_mean=0.62)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] == "alert"


def test_assess_drift_warning_when_confidence_below_acceptable() -> None:
    baseline = make_baseline()
    # Eloszlás OK, de confidence 0.40 (min_acceptable 0.50)
    snapshot = make_snapshot(baseline["label_distribution"], conf_mean=0.40)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] == "warning"
    assert any("confidence" in r.lower() for r in result["reasons"])


def test_assess_drift_warning_when_p95_latency_above_threshold() -> None:
    baseline = make_baseline()
    snapshot = make_snapshot(baseline["label_distribution"], p95=900)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] == "warning"
    assert any("latency" in r.lower() for r in result["reasons"])


def test_assess_drift_returns_insufficient_data_when_window_small() -> None:
    baseline = make_baseline()
    snapshot = make_snapshot(baseline["label_distribution"], n=5)
    result = assess_drift(snapshot, baseline=baseline)
    assert result["status"] == "insufficient_data"


def test_assess_drift_returns_insufficient_data_when_no_predictions() -> None:
    baseline = make_baseline()
    empty = {
        "uptime_seconds": 1.0, "total_requests": 0, "total_errors": 0,
        "recent_window_size": 0, "predictions": None,
    }
    result = assess_drift(empty, baseline=baseline)
    assert result["status"] == "insufficient_data"

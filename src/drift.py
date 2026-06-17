"""Drift detection a /metrics élő adatai és a baseline JSON között.

A baseline a notebook/evaluation.ipynb futtatás eredményeként készült
predikciós eloszlás és confidence-statisztikák. Élesben a /metrics
sliding window-ja szolgál a friss adatként, és három fő drift-jelzőt
számolunk:

1. Label distribution drift (KL-divergencia) — a predikciós címke-eloszlás
   eltérése a baseline-tól.
2. Per-label share drift — bármely címke részarányának abszolút eltérése
   (egyszerű, szabad-szemmel olvasható).
3. Confidence drift — az élő átlagos confidence összehasonlítása a
   baseline átlagával és a min_acceptable_mean-nel.

A státusz három szintű: ok | warning | alert. A küszöböket a
baseline_v1.json drift_thresholds szakasza adja.

A min_observations paraméter biztonsági korlát: kevés adatpont esetén
('warming up') még nem hozunk drift-döntést, hanem 'insufficient_data'-t
adunk vissza.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


DEFAULT_BASELINE_PATH = Path("data/baseline/baseline_v1.json")
MIN_OBSERVATIONS_FOR_DRIFT = 20


def load_baseline(path: Path | str = DEFAULT_BASELINE_PATH) -> dict:
    """Baseline-konfiguráció betöltése JSON-ból."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def kl_divergence(p: dict[str, float], q: dict[str, float], smoothing: float = 1e-6) -> float:
    """KL(P || Q) — két diszkrét eloszlás közötti aszimmetrikus távolság.

    P = friss eloszlás, Q = baseline. Smoothing-gel a 0-osztás elkerülése.
    """
    keys = set(p.keys()) | set(q.keys())
    total = 0.0
    for k in keys:
        p_k = p.get(k, 0.0) + smoothing
        q_k = q.get(k, 0.0) + smoothing
        total += p_k * math.log(p_k / q_k)
    return total


def per_label_share_drift(p: dict[str, float], q: dict[str, float]) -> dict[str, float]:
    """Címkénkénti abszolút eltérés a baseline-tól (százalékpont)."""
    keys = set(p.keys()) | set(q.keys())
    return {k: abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys}


def assess_drift(
    metrics_snapshot: dict,
    baseline: Optional[dict] = None,
    min_observations: int = MIN_OBSERVATIONS_FOR_DRIFT,
) -> dict:
    """Drift-diagnózis a /metrics snapshot és a baseline alapján.

    A visszaadott dict:
      - status: 'ok' | 'warning' | 'alert' | 'insufficient_data'
      - reasons: lista — egy-egy jelző, ami a státuszt indokolja
      - measurements: a tényleges mért értékek (KL, per-label drift, confidence)
      - thresholds: a használt küszöbök (transzparenciáért)
    """
    if baseline is None:
        baseline = load_baseline()

    predictions = metrics_snapshot.get("predictions")
    n = metrics_snapshot.get("recent_window_size", 0)

    if not predictions or n < min_observations:
        return {
            "status": "insufficient_data",
            "reasons": [f"Csak {n} predikció a recent window-ban "
                        f"(minimum {min_observations} szükséges)."],
            "measurements": {},
            "thresholds": baseline.get("drift_thresholds", {}),
        }

    thresholds = baseline["drift_thresholds"]
    reasons: list[str] = []
    severity = "ok"

    def upgrade_severity(new: str) -> None:
        nonlocal severity
        order = {"ok": 0, "warning": 1, "alert": 2}
        if order[new] > order[severity]:
            severity = new

    # 1) Label distribution drift (KL)
    live_labels = predictions["label_distribution"]
    baseline_labels = baseline["label_distribution"]
    kl = kl_divergence(live_labels, baseline_labels)

    if kl >= thresholds["kl_divergence_alert"]:
        reasons.append(
            f"KL-divergencia ({kl:.3f}) átlépte az alert küszöböt "
            f"({thresholds['kl_divergence_alert']})."
        )
        upgrade_severity("alert")
    elif kl >= thresholds["kl_divergence_warning"]:
        reasons.append(
            f"KL-divergencia ({kl:.3f}) átlépte a warning küszöböt "
            f"({thresholds['kl_divergence_warning']})."
        )
        upgrade_severity("warning")

    # 2) Per-label share drift
    share_drift = per_label_share_drift(live_labels, baseline_labels)
    biggest_label, biggest_drift = max(share_drift.items(), key=lambda x: x[1])

    if biggest_drift >= thresholds["label_share_drift_alert"]:
        reasons.append(
            f"'{biggest_label}' részaránya {biggest_drift:.1%} ponttal eltér a "
            f"baseline-tól (alert küszöb: "
            f"{thresholds['label_share_drift_alert']:.0%})."
        )
        upgrade_severity("alert")
    elif biggest_drift >= thresholds["label_share_drift_warning"]:
        reasons.append(
            f"'{biggest_label}' részaránya {biggest_drift:.1%} ponttal eltér a "
            f"baseline-tól (warning küszöb: "
            f"{thresholds['label_share_drift_warning']:.0%})."
        )
        upgrade_severity("warning")

    # 3) Confidence drift
    live_conf_mean = predictions["confidence"]["mean"]
    min_acceptable = baseline["confidence"]["min_acceptable_mean"]
    if live_conf_mean < min_acceptable:
        reasons.append(
            f"Átlagos confidence ({live_conf_mean:.3f}) a min_acceptable_mean "
            f"({min_acceptable:.3f}) alatt."
        )
        upgrade_severity("warning")

    # 4) Latency drift
    live_p95 = predictions["latency_ms"]["p95"]
    p95_threshold = baseline["latency_ms"]["p95_alert_threshold"]
    if live_p95 >= p95_threshold:
        reasons.append(
            f"p95 latency ({live_p95:.0f}ms) átlépte az alert küszöböt "
            f"({p95_threshold}ms)."
        )
        upgrade_severity("warning")

    return {
        "status": severity,
        "reasons": reasons or ["Nincs detektált drift a baseline-hoz képest."],
        "measurements": {
            "kl_divergence": round(kl, 4),
            "biggest_label_share_drift": {
                "label": biggest_label,
                "drift_percentage_points": round(biggest_drift * 100, 2),
            },
            "live_confidence_mean": live_conf_mean,
            "live_p95_latency_ms": live_p95,
            "recent_window_size": n,
        },
        "thresholds": thresholds,
        "baseline_version": baseline.get("version", "unknown"),
    }

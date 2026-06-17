"""Konfigurációs konstansok az article-classifier projekthez.

Ezek a dolgok itt vannak összegyűjtve, hogy egy helyen lássuk őket,
és a többi modul (pipeline, api, monitoring) ezt importálja.
"""

from __future__ import annotations

from pathlib import Path

# --- Útvonalak (a projekt gyökeréhez relatív) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
EVAL_DIR = DATA_DIR / "eval"
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Modell konfiguráció ---
DEFAULT_MODEL = "facebook/bart-large-mnli"
DEFAULT_LABELS_FILE = PROMPTS_DIR / "labels_v1.json"

# A BART-large kontextus mérete. A truncation ennél kicsivel kevesebbre vág
# (hely kell a hipotézis-mondatnak is az NLI input-ban).
MODEL_MAX_TOKENS = 1024
TRUNCATION_TOKEN_BUDGET = 900  # biztonsági ráhagyás

# --- Kód verziója (semantic versioning) ---
# Ez kerül a /version endpoint-ba és a strukturált logokba.
# Szinkronban a Git tag-ekkel (MAJOR.MINOR.PATCH).
CODE_VERSION = "1.0.0"

# --- Confidence küszöbök (monitoringhoz) ---
# Ha a confidence ez alatt van, "low confidence" warning-ot emelünk.
LOW_CONFIDENCE_THRESHOLD = 0.5

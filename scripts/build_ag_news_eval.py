"""
AG News stratifikált eval-set builder.

Letölti az AG News test split-jét (HuggingFace Datasets), token-hossz alapján
3 bin-be osztja (rövid / közepes / hosszú), és kategóriánként + bin-enként
egyenlő számú cikket sample-l. Az eredmény egy CSV: data/eval/ag_news_eval_v1.csv

Reprodukálhatóság:
- A `random_state` rögzítve van (42), így ugyanazt a sample-t kapod minden gépen.
- A bin-határok és sample-méretek konfigurálhatók a script tetején.
- Outputfájl verzióval (v1) — soha NE írd felül; ha új sample kell, készíts v2-t.

Használat (a projekt gyökeréből, aktivált venv-vel):
    python scripts/build_ag_news_eval.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Konfiguráció ---------------------------------------------------------
RANDOM_SEED = 42
MODEL_NAME = "facebook/bart-large-mnli"
DATASET_NAME = "ag_news"
SPLIT = "test"

# Token-hossz binek határai (a BART tokenizer alapján)
LENGTH_BINS = {
    "short":  (0, 40),     # 0-39 token
    "medium": (40, 70),    # 40-69 token
    "long":   (70, 9999),  # 70+ token
}

# Hány cikket sample-lünk kategóriánként és bin-enként
SAMPLES_PER_CELL = 8

# Plusz teljesen véletlen kiegészítés a kerek 100-ig
RANDOM_PADDING = 4

OUTPUT_CSV = Path("data/eval/ag_news_eval_v1.csv")
OUTPUT_META = Path("data/eval/ag_news_eval_v1.meta.json")


def assign_bin(token_count: int) -> str:
    """Egy token-számhoz hozzárendeli a hossz-bint."""
    for bin_name, (lo, hi) in LENGTH_BINS.items():
        if lo <= token_count < hi:
            return bin_name
    raise ValueError(f"Nem értelmezhető hossz: {token_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Csak diagnosztikát ír, nem ír fájlt.")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    print(f"[1/4] AG News '{SPLIT}' split letöltése...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"      Cikk-szám: {len(ds)}")
    print(f"      Kategóriák: {ds.features['label'].names}")

    print(f"\n[2/4] Tokenizer betöltése: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n[3/4] Token-hosszok kiszámítása minden cikkre "
          f"({len(ds)} darab, ez ~30 mp)...")
    # Batch-elt tokenizálás gyorsabb, de a cél itt csak a token-szám.
    token_counts: list[int] = []
    for i, text in enumerate(ds["text"]):
        token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
        if (i + 1) % 1000 == 0:
            print(f"      {i + 1}/{len(ds)}")

    # Csoportosítás (kategória, bin) → indexek
    print(f"\n[4/4] Strat. sampling: {SAMPLES_PER_CELL} / cella + {RANDOM_PADDING} random...")
    label_names = ds.features["label"].names
    cells: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, (label, tc) in enumerate(zip(ds["label"], token_counts)):
        cells[(label, assign_bin(tc))].append(idx)

    # Diagnosztika a cellaméretekről (látjuk-e elég cikket találunk-e mindegyikbe)
    print("\n      Cellaméretek (kategória × hossz-bin):")
    print(f"      {'Kategória':<15s}  " + "  ".join(f"{b:>7s}" for b in LENGTH_BINS))
    for cat_idx, cat_name in enumerate(label_names):
        sizes = [len(cells.get((cat_idx, b), [])) for b in LENGTH_BINS]
        print(f"      {cat_name:<15s}  " + "  ".join(f"{s:>7d}" for s in sizes))

    # Tényleges sampling
    sampled: list[dict] = []
    next_id = 1
    for cat_idx, cat_name in enumerate(label_names):
        for bin_name in LENGTH_BINS:
            pool = cells.get((cat_idx, bin_name), [])
            if len(pool) < SAMPLES_PER_CELL:
                print(f"      ⚠️  Kevés cikk: {cat_name}/{bin_name} = {len(pool)}, "
                      f"de {SAMPLES_PER_CELL} kellene")
            chosen = random.sample(pool, min(SAMPLES_PER_CELL, len(pool)))
            for idx in chosen:
                sampled.append({
                    "id": f"agn_{next_id:03d}",
                    "text": ds[idx]["text"],
                    "true_label": cat_name,
                    "token_count": token_counts[idx],
                    "length_bin": bin_name,
                    "source_index": idx,
                })
                next_id += 1

    # Random padding (egyenletesen az összes kategóriából)
    already_chosen = {row["source_index"] for row in sampled}
    remaining = [i for i in range(len(ds)) if i not in already_chosen]
    pad_chosen = random.sample(remaining, RANDOM_PADDING)
    for idx in pad_chosen:
        sampled.append({
            "id": f"agn_{next_id:03d}",
            "text": ds[idx]["text"],
            "true_label": label_names[ds[idx]["label"]],
            "token_count": token_counts[idx],
            "length_bin": assign_bin(token_counts[idx]),
            "source_index": idx,
        })
        next_id += 1

    df = pd.DataFrame(sampled)
    df = df.drop(columns=["source_index"])  # fejlesztői mező, nem kell a CSV-be

    # Diagnosztika a végső sample-on
    print(f"\n      Végső sample mérete: {len(df)} sor")
    print(f"      Kategória eloszlás: {dict(Counter(df['true_label']))}")
    print(f"      Hossz-bin eloszlás: {dict(Counter(df['length_bin']))}")
    print(f"      Token-hossz min/avg/max: "
          f"{df['token_count'].min()}/{df['token_count'].mean():.0f}/{df['token_count'].max()}")

    if args.dry_run:
        print("\n--dry-run: nem írok fájlt.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ CSV mentve: {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1024:.1f} KB)")

    # Meta-fájl: mi alapján készült (reprodukálhatóság)
    meta = {
        "version": "v1",
        "source_dataset": DATASET_NAME,
        "source_split": SPLIT,
        "tokenizer_model": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "length_bins": {k: list(v) for k, v in LENGTH_BINS.items()},
        "samples_per_cell": SAMPLES_PER_CELL,
        "random_padding": RANDOM_PADDING,
        "total_rows": len(df),
        "label_distribution": dict(Counter(df["true_label"])),
        "bin_distribution": dict(Counter(df["length_bin"])),
        "token_stats": {
            "min": int(df["token_count"].min()),
            "max": int(df["token_count"].max()),
            "mean": float(df["token_count"].mean()),
            "median": float(df["token_count"].median()),
        },
        "build_script": "scripts/build_ag_news_eval.py",
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"✅ Meta mentve: {OUTPUT_META}")


if __name__ == "__main__":
    main()

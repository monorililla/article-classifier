"""
Long-articles eval set builder.

Hosszú (token-szám szerint) cikkekből álló eval set, hogy a pipeline
truncation-viselkedését valós cikkeken tesztelhessük. Két forrásból:

1. BBC News (SetFit/bbc-news) — 5 kategória: entertainment, sport, business,
   politics, tech. Mi ebből 4-et használunk (sport, business, politics,
   technology), a 'tech' címkét a mi 'science and technology'-nkre képezzük.

2. Health-related cikkek — kézzel összeállítva 4 példa (a BBC News-ban nincs
   külön health kategória, és a publikus health-dataset-ek tipikusan
   strukturáltak / nem-hír-jellegűek). A példákat a doksiban hivatkozzuk.

A 'world news' kategóriát ide nem tesszük be — az AG News long bin elég ehhez.

Reprodukálhatóság: random_seed=42, BBC News csak hosszú (>=400 token) cikkekből
sample-l 3-3 cikket kategóriánként.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Konfiguráció ---------------------------------------------------------
RANDOM_SEED = 42
MODEL_NAME = "facebook/bart-large-mnli"
MIN_TOKENS = 400          # csak hosszú cikkeket vegyünk a BBC-ből
SAMPLES_PER_BBC_CATEGORY = 3

# BBC kategória → a mi label-formátumunk
BBC_TO_OUR_LABELS = {
    "sport": "sports",
    "business": "business",
    "politics": "politics",
    "tech": "science and technology",
    # "entertainment" — kihagyjuk, mert nincs nálunk megfelelő címke
}

# Health cikkek kézzel — CDC/WHO közleményekből inspirálva, de saját szövegezéssel.
HEALTH_ARTICLES = [
    {
        "text": (
            "A new study published in the New England Journal of Medicine has revealed that "
            "regular consumption of dietary fiber significantly reduces the risk of cardiovascular "
            "disease in adults over 50. The research, conducted over a decade with more than "
            "100,000 participants, found that those who consumed at least 30 grams of fiber daily "
            "had a 23% lower risk of heart attack and stroke compared to those with low fiber intake. "
            "The lead author, Dr. Sarah Chen of Harvard Medical School, emphasized that fiber-rich "
            "diets including whole grains, legumes, fruits, and vegetables can also lower cholesterol "
            "levels and improve gut microbiome diversity. The findings build on decades of nutritional "
            "research suggesting that plant-based diets play a crucial role in long-term heart health. "
            "Public health officials are calling for updated dietary guidelines based on the results, "
            "noting that the average American consumes only 15 grams of fiber per day — half of the "
            "recommended amount. Researchers are now investigating whether soluble or insoluble fiber "
            "provides greater protection, and whether timing of consumption affects outcomes. "
            "Critics caution that observational studies cannot prove causation, but the consistency of "
            "findings across multiple cohorts makes the link between fiber and heart health "
            "increasingly persuasive."
        ),
        "topic": "cardiovascular health, diet",
    },
    {
        "text": (
            "The World Health Organization issued new guidelines today addressing the global rise in "
            "antibiotic-resistant infections, warning that without urgent action, common medical "
            "procedures could become life-threatening within decades. The report identifies misuse of "
            "antibiotics in both human medicine and livestock farming as the primary drivers of "
            "resistance. According to the WHO, more than 1.27 million deaths globally were directly "
            "attributable to bacterial antimicrobial resistance in 2019, with the highest burden in "
            "low- and middle-income countries. The new guidelines recommend stricter prescribing "
            "practices, expanded surveillance networks, and accelerated development of novel "
            "antibiotics. Health systems are urged to invest in rapid diagnostics, allowing physicians "
            "to identify whether infections are bacterial or viral before prescribing antibiotics. "
            "Hospitals are also advised to strengthen infection control protocols, including hand "
            "hygiene, isolation procedures, and antimicrobial stewardship programs. The pharmaceutical "
            "industry has been criticized for underinvestment in new antibiotic development, citing "
            "low profitability compared to chronic disease medications. Several governments have "
            "announced funding initiatives to support research into next-generation treatments, "
            "including bacteriophage therapy and immunomodulators."
        ),
        "topic": "antibiotic resistance, public health",
    },
    {
        "text": (
            "A breakthrough in mRNA vaccine technology may transform the treatment of certain cancers, "
            "according to clinical trial results presented at the American Society of Clinical "
            "Oncology annual meeting. The personalized vaccines, tailored to each patient's tumor "
            "mutations, showed a 50% reduction in melanoma recurrence when combined with standard "
            "immunotherapy. The trial, involving 157 patients with high-risk stage III or IV melanoma, "
            "demonstrated that the experimental vaccine trained the immune system to recognize and "
            "attack cancer cells based on their unique genetic signatures. Researchers used the same "
            "mRNA platform that proved successful in COVID-19 vaccines, but reprogrammed it to target "
            "tumor-specific neoantigens. Oncologists describe the approach as a potential paradigm "
            "shift in cancer treatment, moving from one-size-fits-all chemotherapy toward truly "
            "personalized therapies. Side effects were generally mild, including fatigue and "
            "injection-site reactions. Larger phase 3 trials are now planned to confirm the results "
            "and explore applications in other cancers including lung, pancreatic, and colorectal. "
            "Regulatory approval, if achieved, could come within three to five years, though "
            "manufacturing scalability and cost remain significant challenges for personalized "
            "vaccines that must be customized for each patient's unique tumor profile."
        ),
        "topic": "cancer immunotherapy, mRNA vaccines",
    },
    {
        "text": (
            "Mental health experts are sounding alarm bells over the growing impact of social media "
            "on adolescent psychology, with new longitudinal data showing significant increases in "
            "anxiety and depression among teenagers who spend more than three hours daily on "
            "platforms like Instagram, TikTok, and Snapchat. The American Academy of Pediatrics "
            "published a comprehensive review of 47 studies, concluding that excessive social media "
            "use is associated with poorer sleep quality, lower self-esteem, and increased rates of "
            "self-harm, particularly among adolescent girls. The report recommends that parents "
            "delay smartphone introduction until age 14 and social media access until 16, while "
            "schools should implement device-free policies during the school day. Tech companies "
            "have faced increasing pressure from lawmakers and advocacy groups to redesign their "
            "platforms to reduce addictive features such as infinite scrolling, push notifications, "
            "and engagement-maximizing algorithms targeted at young users. Some platforms have "
            "introduced screen-time limits and content moderation tools, but critics argue these "
            "measures are insufficient. Pediatricians are also calling for expanded mental health "
            "services in schools and improved training for primary care providers to identify and "
            "treat anxiety and depression in young patients earlier, before symptoms become severe."
        ),
        "topic": "adolescent mental health, social media",
    },
]

OUTPUT_CSV = Path("data/eval/long_articles_eval_v1.csv")
OUTPUT_META = Path("data/eval/long_articles_eval_v1.meta.json")


def main() -> None:
    random.seed(RANDOM_SEED)

    print("[1/3] BBC News dataset letöltése...")
    ds = load_dataset("SetFit/bbc-news", split="test")
    print(f"      Cikk-szám: {len(ds)}")

    print(f"\n[2/3] Tokenizer betöltése: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\n[3/3] Long cikkek sample-lése (>= {MIN_TOKENS} token)...")
    rows: list[dict] = []
    next_id = 1

    # BBC News-ból
    for bbc_label, our_label in BBC_TO_OUR_LABELS.items():
        # Filtered pool: csak ezt a kategóriát + hosszú cikkek
        pool: list[tuple[int, int]] = []  # (idx, token_count)
        for idx in range(len(ds)):
            if ds[idx]["label_text"] != bbc_label:
                continue
            tc = len(tokenizer.encode(ds[idx]["text"], add_special_tokens=False))
            if tc >= MIN_TOKENS:
                pool.append((idx, tc))

        print(f"      BBC '{bbc_label}': {len(pool)} hosszú cikk a pool-ban")
        chosen = random.sample(pool, min(SAMPLES_PER_BBC_CATEGORY, len(pool)))
        for idx, tc in chosen:
            rows.append({
                "id": f"long_{next_id:03d}",
                "text": ds[idx]["text"],
                "true_label": our_label,
                "token_count": tc,
                "source": "bbc-news",
            })
            next_id += 1

    # Health cikkek (kézzel)
    print(f"      Health cikkek: {len(HEALTH_ARTICLES)} (kézzel)")
    for article in HEALTH_ARTICLES:
        tc = len(tokenizer.encode(article["text"], add_special_tokens=False))
        rows.append({
            "id": f"long_{next_id:03d}",
            "text": article["text"],
            "true_label": "health",
            "token_count": tc,
            "source": "manual",
        })
        next_id += 1

    df = pd.DataFrame(rows)

    print(f"\n      Végső sample mérete: {len(df)} sor")
    print(f"      Kategória eloszlás: {dict(Counter(df['true_label']))}")
    print(f"      Token-hossz min/avg/max: "
          f"{df['token_count'].min()}/{df['token_count'].mean():.0f}/{df['token_count'].max()}")
    print(f"      >= 1024 token (truncation szükséges): "
          f"{(df['token_count'] >= 1024).sum()}/{len(df)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ CSV mentve: {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size / 1024:.1f} KB)")

    meta = {
        "version": "v1",
        "purpose": "Truncation-viselkedés tesztelése valódi hosszú cikkeken",
        "sources": {
            "bbc-news": {
                "dataset": "SetFit/bbc-news",
                "split": "test",
                "min_tokens": MIN_TOKENS,
                "samples_per_category": SAMPLES_PER_BBC_CATEGORY,
                "categories_used": list(BBC_TO_OUR_LABELS.keys()),
                "category_mapping": BBC_TO_OUR_LABELS,
            },
            "manual": {
                "purpose": "Health kategória — BBC News-ban és AG News-ban nincs",
                "count": len(HEALTH_ARTICLES),
                "topics": [a["topic"] for a in HEALTH_ARTICLES],
            },
        },
        "tokenizer_model": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "total_rows": len(df),
        "label_distribution": dict(Counter(df["true_label"])),
        "source_distribution": dict(Counter(df["source"])),
        "token_stats": {
            "min": int(df["token_count"].min()),
            "max": int(df["token_count"].max()),
            "mean": float(df["token_count"].mean()),
            "above_1024": int((df["token_count"] >= 1024).sum()),
        },
        "build_script": "scripts/build_long_articles_eval.py",
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"✅ Meta mentve: {OUTPUT_META}")


if __name__ == "__main__":
    main()

# Article Classifier — Zero-Shot News Categorization Pipeline

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Status](https://img.shields.io/badge/status-WIP-orange)]()

Egyetemi beadandó projekt: angol nyelvű hírek/cikkek automatikus kategorizálása
**zero-shot szöveg-osztályozással** a HuggingFace `facebook/bart-large-mnli` modell
segítségével. Nem igényel címkézett training adatot, új kategória hozzáadása
csak konfiguráció-módosítás.

## ✨ Főbb jellemzők

- 🤖 **Zero-shot classification** — nincs szükség saját training datasetre
- 🚀 **REST API** FastAPI-val, OpenAPI/Swagger dokumentációval
- 🐳 **Konténerizált** — egy parancs az indításhoz (`docker compose up`)
- 📓 **Futtatható Jupyter notebook** a végponttól végpontig demóval
- 📊 **Beépített monitoring** — strukturált loggolás, latency / confidence metrikák, drift detection
- 🔖 **Verziózott** — kód, modell, címkék és eval-dataset is

## 📋 Tartalomjegyzék

- [Architektúra](docs/architecture.md)
- [Adathalmaz dokumentáció](docs/dataset.md)
- [Monitoring és KPI-ok](docs/monitoring.md)
- [Verziózási stratégia](docs/versioning.md)
- [API referencia](docs/api.md)

## 🚀 Gyors start

### Helyi futtatás (fejlesztéshez)

```bash
# 1. Klón
git clone https://github.com/monorililla/article-classifier.git
cd article-classifier

# 2. Virtuális környezet
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Függőségek
pip install -r requirements.txt

# 4. Notebook indítása
jupyter lab notebook/exploration.ipynb
```

### Docker (produktív futtatás)

```bash
docker compose up --build
# API elérhető: http://localhost:8000
# Swagger UI:   http://localhost:8000/docs
```

### Példa API-hívás

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple unveiled its new M5 chip today, claiming a 40% performance boost.",
    "labels": ["technology", "sports", "politics", "business"]
  }'
```

Válasz:
```json
{
  "predicted_label": "technology",
  "confidence": 0.94,
  "all_scores": {
    "technology": 0.94,
    "business": 0.04,
    "politics": 0.01,
    "sports": 0.01
  },
  "model_version": "facebook/bart-large-mnli",
  "request_id": "..."
}
```

## 🏗️ Mappastruktúra

```
article-classifier/
├── docs/         # Részletes dokumentáció (architektúra, monitoring, ...)
├── notebook/     # Futtatható Jupyter notebook
├── data/         # Eval dataset és címke-konfigurációk (verziózva)
├── src/          # Forráskód: pipeline, API, monitoring
├── tests/        # Unit és integrációs tesztek
├── logs/         # Runtime monitoring logok (gitignore-olva)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 🧪 Tesztek futtatása

```bash
pytest tests/ -v
```

## 📚 Háttér

A modell a [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli)
adathalmazon (~412k mondatpár) lett tanítva NLI feladatra.
Részletes magyarázat a zero-shot megközelítésről:
[docs/architecture.md](docs/architecture.md).

## 📝 Licenc

MIT — szabadon használható oktatási és kutatási célra.

---

**Készítette:** Mónor Lilla
**Egyetem:** Corvinus Egyetem
**Tárgy:** [tárgynév]
**Év:** 2026

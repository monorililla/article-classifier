# Article Classifier

Zero-shot angol nyelvű hírcikk-osztályozó REST API a HuggingFace
`facebook/bart-large-mnli` modell köré építve. Címkézett training-adat
nélkül kategorizál, új kategóriák felvétele konfigurációs változás.

## Funkciók

- REST API (FastAPI) — `/classify`, `/health`, `/version`, `/metrics`,
  `/metrics/drift`
- Strukturált JSON Lines logging (PII-mentes, hash-elt input)
- In-memory metrika-gyűjtő sliding window-nal
- Drift detection KL-divergencia és per-label share alapján
- Bemeneti adatminőség-ellenőrzés (rövid input, nem-angol, HTML)
- Truncation kezelés a 1024-token limit fölött
- Multi-stage Docker build, modell pre-downloadolva
- 30 pytest teszt (tesztek a pipeline-ra, API-ra, logger-re, drift-re,
  quality-re)

## Gyors start

### Helyi futtatás

```bash
git clone https://github.com/monorililla/article-classifier.git
cd article-classifier

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn src.api:app --reload
```

A modell betöltése Apple Silicon-on (MPS) ~5 másodperc, predikció ~250 ms.

### Docker

```bash
docker compose up
```

Az API a `http://localhost:8000`-en érhető el. CPU-only Linux konténerben
a latency nagyobb (1100-1700 ms predikciónként).

### Példa request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple unveiled the new M5 chip today."}'
```

Válasz:
```json
{
  "predicted_label": "science and technology",
  "confidence": 0.31,
  "all_scores": {...},
  "truncated": false,
  "input_token_count": 8,
  "model_version": "facebook/bart-large-mnli",
  "labels_version": "v1",
  "code_version": "0.2.0",
  "latency_ms": 1091.9,
  "timestamp": "2026-06-17T16:39:35.857Z",
  "request_id": "d5356e09-..."
}
```

## Interaktív API dokumentáció

Futás közben:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Dokumentáció

| Doksi | Mit tartalmaz |
|---|---|
| [architecture.md](docs/architecture.md) | Projekt áttekintés, NLI-trükk, komponens-struktúra, deployment-modellek |
| [dataset.md](docs/dataset.md) | MultiNLI training-adat, AG News eval, long-articles set, label-set |
| [monitoring.md](docs/monitoring.md) | KPI-ok, drift-küszöbök, retraining flow, dashboard mockup |
| [versioning.md](docs/versioning.md) | Mit hogyan verziózunk: kód, modell, prompt, eval data |
| [api.md](docs/api.md) | Endpoint-ok részletes referenciája, példa-request-ek |

## Repo-struktúra

```
article-classifier/
├── src/                  Forráskód (api, pipeline, monitoring, drift, quality)
├── tests/                Pytest tesztek
├── data/
│   ├── prompts/          Verziózott label-set-ek
│   ├── eval/             Verziózott eval CSV-k és meta-fájlok
│   └── baseline/         Drift-detection referencia
├── notebook/             Jupyter notebook-ok (exploration, evaluation)
├── scripts/              Reprodukálható dataset-építők
├── docs/                 Részletes dokumentáció
├── logs/                 Runtime predikciós logok (JSON Lines)
├── Dockerfile            Multi-stage build
├── docker-compose.yml    Egy-paranccsal indítás
└── requirements.txt      Pinneolt függőségek
```

## Tesztek

```bash
pytest
```

A pipeline-tesztek a tényleges modellt töltik (~10 mp a teljes futás).
Az API-, logger-, drift- és quality-tesztek modell-mentesek (mock
classifierrel), <1 mp alatt lefutnak.

## Notebook-ok

Két Jupyter notebook a `notebook/` mappában:

- `exploration.ipynb` — felfedező notebook: a HF zero-shot pipeline
  alapfunkcionalitása, truncation viselkedés-tesztelés.
- `evaluation.ipynb` — kiértékelés mindkét eval-seten (AG News + long-articles),
  per-class F1, confusion matrix, per-hossz-bin metrikák, baseline-számokhoz
  vezető teljes futtatás.

A notebook futtatása:
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=article-classifier
jupyter lab notebook/evaluation.ipynb
```

## Licenc

MIT (oktatási és kutatási célra).

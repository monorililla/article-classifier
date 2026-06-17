# Monitoring és KPI-ok

## Áttekintés

A rendszer három különálló monitoring-réteget használ, mindegyiknek
külön felelőssége és időskálája:

| Réteg | Forrás | Időskála | Mit ad? |
|---|---|---|---|
| Strukturált log | `logs/predictions.jsonl` | request-szintű | Audit trail, debug |
| Live metrika | `GET /metrics` | utolsó 1000 request | Sliding window summary |
| Drift assessment | `GET /metrics/drift` | live vs baseline | Szöveges drift-jelzés |

## Strukturált predikciós log

Minden `/classify` hívás generál egy soros JSON rekordot a
`logs/predictions.jsonl` fájlba (JSON Lines formátum).

A rekord struktúrája:
```json
{
  "timestamp": "2026-06-17T16:39:34.741Z",
  "request_id": "9189554d-6b6c-494c-b431-88374194d00a",
  "client_id": null,
  "input": {
    "char_count": 91,
    "token_count": 18,
    "hash16": "a3c8f1d20b7e4682",
    "truncated": false,
    "processed_token_count": 18
  },
  "prediction": {
    "label": "sports",
    "confidence": 0.7918,
    "all_scores": {
      "sports": 0.7918,
      "world news": 0.0793,
      "business": 0.0557,
      "health": 0.0319,
      "science and technology": 0.0222,
      "politics": 0.0192
    }
  },
  "performance": {"latency_ms": 1692.9},
  "version": {
    "model": "facebook/bart-large-mnli",
    "labels": "v1",
    "code": "1.0.0"
  }
}
```

A formátum **streamelve feldolgozható** (`tail -f logs/predictions.jsonl`),
és bevihető bármely modern log-aggregátorba (Loki, ELK, CloudWatch,
Datadog Logs). Egy egyszerű elemzés `pandas`-szal:

```python
import pandas as pd
df = pd.read_json("logs/predictions.jsonl", lines=True)
df = pd.json_normalize(df.to_dict("records"))
df.groupby("prediction.label")["prediction.confidence"].mean()
```

### PII / GDPR

A logba **nem kerül az input szöveg** — csak annak SHA-256 hash-ének
első 16 karaktere. Ennek két célja:

1. Visszakereshetőség: ha egy konkrét szöveg újra megjelenik, a hash
   alapján beazonosítható (egyezés-keresés).
2. Adatvédelem: a tartalom nem rekonstruálható a logból, így a log
   szabadon archiválható, megosztható, kompromittálódás esetén nincs
   tartalom-szivárgás.

A `tests/test_prediction_logger.py` egy regression-teszttel garantálja:
ha a kód jövőbeli változata véletlenül szivárogtatná az input-szöveget,
a teszt elhasal.

## Live metrikák (`GET /metrics`)

Egy memóriában tartott sliding window (capacity=1000) az utolsó N
predikció statisztikáit aggregálja. A snapshot:

```json
{
  "uptime_seconds": 46.2,
  "total_requests": 2,
  "total_errors": 0,
  "recent_window_size": 2,
  "predictions": {
    "label_distribution": {"sports": 0.5, "science and technology": 0.5},
    "confidence": {
      "mean": 0.5497,
      "min": 0.3076,
      "max": 0.7918,
      "low_count": 1,
      "low_ratio": 0.5
    },
    "latency_ms": {
      "mean": 1392.4, "p50": 1392.4, "p95": 1662.9, "p99": 1686.9
    },
    "input_tokens": {"mean": 17.5, "max": 18},
    "truncation": {"count": 0, "ratio": 0.0}
  }
}
```

A `MetricsCollector` thread-safe (mutex zárolás), így multi-worker
uvicorn esetén is konzisztens marad.

## KPI-ok és drift küszöbök

Az alábbi tábla a baseline-mérés (`data/baseline/baseline_v1.json`) és
a drift-modul küszöbei alapján:

| KPI | Baseline | Warning | Alert | Forrás |
|---|---|---|---|---|
| Accuracy (eval seten) | 55% | < 50% | < 40% | `notebook/evaluation.ipynb` |
| Átlagos confidence | 0.61 | < 0.50 | < 0.40 | live `/metrics` mean |
| `low_ratio` (conf < 0.5 aránya) | ~30% | > 50% | > 70% | live `/metrics` |
| Címke-eloszlás KL-divergencia | 0.0 | >= 0.15 | >= 0.30 | live vs baseline |
| Per-label share drift | 0.0 | >= 10%-pont | >= 20%-pont | live vs baseline |
| p95 latency (CPU) | 460 ms | >= 700 ms | >= 1500 ms | live `/metrics` |
| Truncation ratio | 0% | > 5% | > 20% | live `/metrics` |
| Error rate (5xx) | 0% | > 1% | > 5% | live `/metrics` total_errors |

A KL-divergencia értelmezése: egy információelméleti távolság-mérőszám
két diszkrét eloszlás között. 0 = azonosak, magasabb érték = nagyobb
eltérés. Smoothing a 0-osztás elkerülésére (1e-6 minden label-en).

## Adatminőségi problémák

Az input-validáció két szinten történik:

### Pydantic / pipeline szintű (fatal, 422)
A request elutasításra kerül, a metrika `total_errors`-ba kerül:
- Üres string vagy csak whitespace
- > 50 000 karakter (Pydantic max_length)
- Hiányzó `text` mező

### Heurisztikus szintű (non-fatal, warning a logba)
A request feldolgozásra kerül, de a logba `data_quality_warning`-ot
írunk (`src/quality.py`):

| Issue | Detektálás | Indok |
|---|---|---|
| `input_too_short` | < 50 karakter | A modell pontossága rohamosan romlik |
| `possibly_non_english` | > 10% nem-ASCII karakter | A modell csak angolra van kalibrálva |
| `html_tags_detected` | >= 3 HTML tag | Cleanup javasolt, mert a tag-ek tokenként számolódnak |

Ezek a `prediction_logger.log_data_quality_warning()` hívások a
`logs/predictions.jsonl`-be `WARNING` szintű rekordként kerülnek.
Production-ben egy log-aggregátor riasztást emelhet, ha pl. az
`input_too_short` aránya hirtelen 5%-ról 20%-ra ugrik — ez upstream
input-distribution drift jelzése (pl. kliens-bug, ami csak headline-t
küld lead nélkül).

## Drift assessment (`GET /metrics/drift`)

A drift-endpoint a baseline_v1.json és a recent /metrics adatait veti
össze, és négy szintű státuszt ad:

- **`insufficient_data`**: a recent window kisebb, mint 20 — még nem
  hozunk drift-döntést.
- **`ok`**: minden mért érték a küszöbök alatt.
- **`warning`**: legalább egy érték a warning küszöb felett. Még nem
  igényel azonnali beavatkozást, de figyelendő.
- **`alert`**: legalább egy érték az alert küszöb felett. Beavatkozás
  szükséges (lásd lent: retraining trigger flow).

A válasz tartalmaz `reasons` (szöveges indoklás), `measurements`
(tényleges mért értékek), és `thresholds` (a használt küszöbök) mezőket
— transzparenciáért, hogy ne legyen "fekete-doboz" döntés.

## Retraining trigger

Mivel a modell zero-shot, a hagyományos "retraining" itt nem klasszikus
fine-tuning. A trigger-flow:

1. **Detektálás**: a `/metrics/drift` `alert` státuszt ad, vagy a
   strukturált logban tartós `low_confidence` mintázat látható.
2. **Diagnózis**: az ops-csapat megvizsgálja a `reasons`-t. Pl.
   "biggest_label_share_drift: politics +25%-pont" → új politikai
   esemény (választás, háború) miatt nőtt a politikai cikkek aránya.
3. **Beavatkozás opciók** (a diagnózistól függő sorrendben):
   1. **Új baseline mérése**: ha a drift "új normális" (pl. választás
      után tartós), a baseline_v1.json → baseline_v2.json. A
      `notebook/evaluation.ipynb`-t lefuttatjuk a friss eval-seten,
      a baseline_v2.json az új küszöbökkel készül.
   2. **Új címke-set**: ha a drift új téma miatt van (pl. új sport-ág
      kategória), a `labels_v1.json` → `labels_v2.json`-be új címkék
      kerülnek. A pipeline újraindítása után az új címkék azonnal
      használhatók (zero-shot előny).
   3. **Modell-csere**: ha a label-eloszlás OK, de a confidence tartósan
      alacsony, érdemes másik modellel próbálkozni (pl.
      `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
      jobb teljesítményt mutat zero-shot-on egyes benchmark-okon).
   4. **Fine-tuning**: hosszabb távú megoldás, ha az accuracy a
      `low_acceptable` szint alá esik tartósan. Ez a projekt jelenlegi
      hatálya kívül esik, de a `src/pipeline.py` interfész változatlanul
      maradna — az `ArticleClassifier` osztály bármely zero-shot képes
      modellt becsomagolja.

A retraining-trigger döntéseket Git-be commitoljuk:
- Új baseline → új commit a `data/baseline/baseline_v*.json`-ról.
- Új címke-set → új commit a `data/prompts/labels_v*.json`-ról.
- Modell-csere → új commit a `requirements.txt` és `src/config.py`
  `DEFAULT_MODEL` konstansra.

Így minden retraining-döntés **olvasható verzió-történetben** marad meg,
és bármikor visszafordítható (Git revert).

## Dashboard mockup (verbális leírás)

A jelenlegi `/metrics` és `/metrics/drift` endpoint-ok JSON-t adnak
vissza. Production-ben egy egyszerű Grafana dashboard a következő
panelekkel kötendő be:

1. **Request volume** — `total_requests` időbeli alakulása (1 órás bin-ek)
2. **Confidence trend** — `predictions.confidence.mean` időbeli görbe
3. **Label distribution** — `predictions.label_distribution` stacked area
4. **Latency p50/p95/p99** — három görbe egy panelen
5. **Drift status** — szöveges állapot-jelző (zöld/sárga/piros)
6. **Error rate** — `total_errors / total_requests` arány

A Grafana-Loki integráció a `logs/predictions.jsonl`-en keresztül a
strukturált logokat is keresőzhetővé teszi.

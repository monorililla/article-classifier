# API referencia

Az `article-classifier` REST API minden endpoint-ja JSON-on keresztül
kommunikál. A teljes interaktív dokumentáció a Swagger UI-on érhető el
futás közben:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Endpoint-ok

### `POST /classify`

Egy cikk osztályozása.

**Request body:**
```json
{
  "text": "Apple unveiled its new M5 chip today.",
  "labels": null,
  "request_id": null
}
```

| Mező | Típus | Kötelező? | Leírás |
|---|---|---|---|
| `text` | string (1-50000 char) | igen | Az osztályozandó szöveg |
| `labels` | list[string] | nem | Felülbírálja a default címke-listát |
| `request_id` | string | nem | Egyedi azonosító (auto-generált, ha hiányzik) |

**Sikeres válasz (200):**
```json
{
  "predicted_label": "science and technology",
  "confidence": 0.3076,
  "all_scores": {
    "science and technology": 0.3076,
    "business": 0.2721,
    "world news": 0.2594,
    "health": 0.0689,
    "sports": 0.0483,
    "politics": 0.0437
  },
  "truncated": false,
  "input_token_count": 17,
  "processed_token_count": 17,
  "model_version": "facebook/bart-large-mnli",
  "labels_version": "v1",
  "code_version": "0.2.0",
  "latency_ms": 1091.9,
  "timestamp": "2026-06-17T16:39:35.857Z",
  "request_id": "d5356e09-f480-4b94-a758-c30ed8ddf872"
}
```

**Hibák:**
- `422 Unprocessable Entity` — érvénytelen input (üres szöveg, túl hosszú,
  hiányzó mező)
- `500 Internal Server Error` — váratlan modell- vagy szerver-hiba

**Példák:**

Default label-set:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "The Lakers defeated the Celtics 112-104 tonight."}'
```

Saját label-set:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The chef seasoned the risotto with saffron.",
    "labels": ["cooking", "music", "automotive", "fashion"]
  }'
```

Egyedi `request_id` (visszakereshetőséghez):
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple unveiled its new M5 chip.",
    "request_id": "demo-123"
  }'
```

---

### `GET /health`

Healthcheck. A Docker `HEALTHCHECK` directiva ezt használja.

**Válasz (200):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "code_version": "0.2.0",
  "labels_version": "v1"
}
```

A `status` mező értékei:
- `"ok"` — a modell betöltve, az API kész requesteket fogadni
- `"loading"` — a modell még tölt be (lifespan-startup folyamatban)

---

### `GET /version`

A futó komponensek verzió-információi.

**Válasz (200):**
```json
{
  "code_version": "0.2.0",
  "model_name": "facebook/bart-large-mnli",
  "labels_version": "v1",
  "labels": [
    "world news",
    "sports",
    "business",
    "science and technology",
    "politics",
    "health"
  ],
  "device": "cpu",
  "model_max_tokens": 1024,
  "truncation_token_budget": 900
}
```

A `device` mező ("cpu", "mps", "cuda") attól függ, hol fut a szolgáltatás:
helyi macOS-en `mps`, Docker-konténerben `cpu`, GPU-s deploymentben `cuda`.

---

### `GET /metrics`

Aggregált statisztikák a recent predikciókról (sliding window, capacity=1000).

**Válasz (200) — üres állapot:**
```json
{
  "uptime_seconds": 1.2,
  "total_requests": 0,
  "total_errors": 0,
  "recent_window_size": 0,
  "predictions": null
}
```

**Válasz (200) — predikciókkal:**
```json
{
  "uptime_seconds": 46.2,
  "total_requests": 2,
  "total_errors": 0,
  "recent_window_size": 2,
  "predictions": {
    "label_distribution": {
      "sports": 0.5,
      "science and technology": 0.5
    },
    "confidence": {
      "mean": 0.5497,
      "min": 0.3076,
      "max": 0.7918,
      "low_count": 1,
      "low_ratio": 0.5
    },
    "latency_ms": {
      "mean": 1392.4,
      "p50": 1392.4,
      "p95": 1662.9,
      "p99": 1686.9
    },
    "input_tokens": {"mean": 17.5, "max": 18},
    "truncation": {"count": 0, "ratio": 0.0}
  }
}
```

A részletekért lásd: [`monitoring.md`](monitoring.md).

---

### `GET /metrics/drift`

Drift-státusz a baseline (`data/baseline/baseline_v1.json`) és a recent
`/metrics` adatok összevetéséből.

**Válasz (200) — kevés adat:**
```json
{
  "status": "insufficient_data",
  "reasons": ["Csak 5 predikció a recent window-ban (minimum 20 szükséges)."],
  "measurements": {},
  "thresholds": {
    "kl_divergence_warning": 0.15,
    "kl_divergence_alert": 0.30,
    "label_share_drift_warning": 0.10,
    "label_share_drift_alert": 0.20
  }
}
```

**Válasz (200) — minden rendben:**
```json
{
  "status": "ok",
  "reasons": ["Nincs detektált drift a baseline-hoz képest."],
  "measurements": {
    "kl_divergence": 0.0432,
    "biggest_label_share_drift": {
      "label": "sports",
      "drift_percentage_points": 4.2
    },
    "live_confidence_mean": 0.61,
    "live_p95_latency_ms": 460,
    "recent_window_size": 100
  },
  "thresholds": {...},
  "baseline_version": "v1"
}
```

**Válasz (200) — alert:**
```json
{
  "status": "alert",
  "reasons": [
    "KL-divergencia (0.412) átlépte az alert küszöböt (0.30).",
    "'politics' részaránya 25.0% ponttal eltér a baseline-tól (alert küszöb: 20%)."
  ],
  "measurements": {...},
  "thresholds": {...},
  "baseline_version": "v1"
}
```

Státusz-értékek: `insufficient_data | ok | warning | alert | baseline_missing`.

A részletekért lásd: [`monitoring.md`](monitoring.md).

---

## CORS és authentikáció

A jelenlegi konfigurációban **nincs CORS-szabály és nincs authentikáció**
— a deployment-modellt feltételezi, hogy az API egy zárt hálózaton fut
(VPC, intranet) vagy egy külső gateway szűri a bejövő forgalmat.

Production-deploymentnél javasolt:

- **API key middleware** — a `Authorization: Bearer <key>` header
  ellenőrzése egy FastAPI dependency-vel.
- **CORS konfiguráció** — explicit `allow_origins` lista a
  `fastapi.middleware.cors.CORSMiddleware`-ben.
- **Rate limiting** — pl. `slowapi` library, IP-alapú limitálás.

Ezek a bővítések nem érintik a meglévő endpoint-szerződést.

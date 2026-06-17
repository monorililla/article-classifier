# Verziózási stratégia

Mit verziózunk és hol — átfogó kép a projekt minden artifaktumáról.

## A négy verziózott artifakt

| Artifakt | Hol van? | Hogyan azonosítjuk? |
|---|---|---|
| Forráskód | Git repo | Git commit hash + tag (semver) |
| Modell | HuggingFace Hub | `model_name` + revision hash |
| Címke-set / prompt | `data/prompts/labels_vN.json` | Fájlnévben szereplő `vN` |
| Eval / baseline data | `data/eval/*.csv`, `data/baseline/*.json` | Fájlnévben szereplő `vN` |

A négy artifakt **független verziójú**: új kód-verzió nem jelent automatikusan
új modellt, és új címke-set nem jelent új kódot. A `src/config.py`
`CODE_VERSION` és a `labels_v1.json` `version` mezője külön mozognak,
és minden `/classify` válasz mindkettőt visszaadja.

## Forráskód verziózás (Git + Conventional Commits)

A repo Conventional Commits-stílusú üzeneteket használ:

```
<type>(<scope>): <subject>
<body — opcionális>
```

| Type | Mire való |
|---|---|
| `feat` | Új funkció (új endpoint, új metrika, új modul) |
| `fix` | Bug-javítás |
| `docs` | Csak dokumentációs változás (.md fájlok) |
| `refactor` | Belső átszervezés, viselkedés-változatlan |
| `test` | Csak tesztek hozzáadása / módosítása |
| `chore` | Kódbázis-takarítás (komment-csere, .gitignore stb.) |

Példa:

```
feat(monitoring): drift detection module and /metrics/drift endpoint

src/drift.py — analitikai modul:
- KL-divergencia a live label-eloszlás és a baseline között
- Per-label share drift (egyszerű, abszolút %-pont eltérés)
...
```

A scope a leginkább érintett modul (`api`, `pipeline`, `monitoring`,
`docker`, `notebook`). Ez segít a `git log --grep="^feat(monitoring)"`
típusú szűrésekben, és a CHANGELOG-generálásban.

### Branch-stratégia

A `main` branch mindig deployolható állapotú. A munka feature branch-eken
zajlik:

```
main                       ← deployolható
├── feature/dataset-and-pipeline
├── feature/api-and-docker
└── feature/monitoring-and-docs
```

A branch befejezése után `--no-ff` merge a `main`-be, így a merge-commit
láthatóvá teszi, hol kezdődött és hol végződött egy feature.

```
git checkout -b feature/<name>
# ... commit-ok ...
git checkout main
git merge feature/<name> --no-ff -m "Merge feature/<name>"
git push
```

### Tag-ek (mérföldkövek)

A tag-eket nem minden commitra adunk, hanem tényleges deployolt
mérföldkövekre. Az aktuális tag-történet:

- `v1.0.0` — a teljes csomag (pipeline, REST API, Docker, monitoring,
  drift detection, dokumentáció). Az első deployolható release.

A `CODE_VERSION` a `src/config.py`-ban szinkronban van a tag-ekkel
(SemVer: MAJOR.MINOR.PATCH). Új feature → MINOR bump, breaking change
(pl. `/classify` válasz-séma változás) → MAJOR bump.

## Modell-verziózás

A HuggingFace Hub minden modell-revízión `commit hash`-et tart fenn.
A `requirements.txt` pinneolja a `transformers` verziót, és a kód a
`facebook/bart-large-mnli` modell-azonosítót **konstansként** használja
(`src/config.py` `DEFAULT_MODEL`).

Minden `/classify` válasz tartalmazza:
```json
"version": {"model": "facebook/bart-large-mnli", ...}
```

Ha pontosabb pinning kell (pl. egy konkrét HF revision), ez egy egyszerű
bővítés:

```python
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
    revision="d7645e127eaf1aefc7862fd59a17a5aa8558b8ce",  # példa
)
```

A `transformers` library ekkor pontosan azt a revíziót tölti le. A
projekt jelenlegi formájában a HF "main" revision-jét használjuk
(implicit `revision="main"`), ami a HF-en ritkán változik egy lefagyott
modell esetén.

## Címke-set / prompt verziózás

A `data/prompts/labels_v1.json` mind a candidate labels listát, mind a
hipotézis-templatet tartalmazza. Új verzió-fájl ezekben az esetekben:

| Változás | Új fájl |
|---|---|
| Új címke felvétele | `labels_v2.json` |
| Címke átnevezése (pl. `health` → `healthcare`) | `labels_v2.json` |
| Hipotézis-template csere | `labels_v2.json` |
| Modell-csere (más modellnek más optimális template lehet) | `labels_v2.json` |

A régi fájl **soha nincs felülírva** — bármely korábbi futtatás
reprodukálható. A `src/config.py` `DEFAULT_LABELS_FILE` konstans állítja
be, melyik aktuális. A pipeline képes runtime-ban más labels-fájlt
betölteni:

```python
clf = ArticleClassifier(labels_file="data/prompts/labels_v2.json")
```

## Eval / baseline data verziózás

Az `data/eval/ag_news_eval_v1.csv` és a `data/baseline/baseline_v1.json`
ugyanezt a "soha nem mutáljuk, új verzió esetén új fájl" mintát követik.

A baseline-fájlt **a baseline-mérés egy konkrét futtatása** generálja.
Új verzió szükséges, ha:

- Új modell-verzió kerül használatba (más eloszlás, más latency).
- Az eval set bővül (új cikkek, vagy nagyobb sample).
- Új küszöbök (pl. szigorúbb drift-detektálás).

A `baseline_v1.json` `source` mezője dokumentálja, melyik eval-futtatás
generálta:
```json
"source": "notebook/evaluation.ipynb on AG News stratifikált eval set (100 sor)"
```

## Retraining flow röviden

(A részletek a `docs/monitoring.md`-ban.)

A zero-shot architektúrában a "retraining" négy lehetséges válasz:

1. **Új baseline** — a drift "új normális", új küszöbök kellenek.
2. **Új címke-set** — új téma jelent meg.
3. **Modell-csere** — másik HF modellt választunk.
4. **Fine-tuning** — utolsó opció, ha minden más kevés.

Mind a négy **Git-commit eseménynek** számít, vagyis a retraining nyoma
mindig benne van a verzió-történetben:

```
$ git log --oneline data/baseline/
```

## Reprodukálhatóság

Egy adott időpontban érvényes futtatás reprodukálásához három dolog
elég:

1. **Git commit hash** (kód-verzió)
2. **`requirements.txt`** (a függőségek verziói pinneolva, a `transformers`
   és a `torch` is konkrét verzióhoz kötve)
3. **A `data/prompts/labels_*.json` és `data/baseline/baseline_*.json`**
   verzió-fájlok, amik az adott commit-ban érvényesek

A `git checkout <commit_hash>` után a `pip install -r requirements.txt`
+ `notebook/evaluation.ipynb` újrafuttatása biztosan ugyanazt az
eredményt adja (a HF model revision konzisztenciájának függvényében).

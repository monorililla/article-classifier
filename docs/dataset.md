# Adathalmazok

Három különböző adathalmaz játszik szerepet a projektben. Egyik sem
training-adat — a modell pre-trained, nincs fine-tuning. A három halmaz
funkciója: a **modell training-forrásának dokumentálása** (transzparencia),
a **kiértékelés alapja** (baseline), és a **truncation-edge-case tesztelése**.

## 1. A pre-trained modell training-adata: MultiNLI

Forrás: [`nyu-mll/multi_nli`](https://huggingface.co/datasets/nyu-mll/multi_nli)
(412 349 mondatpár, angol nyelvű)

A `facebook/bart-large-mnli` modellt erre a korpuszra tanították NLI
(Natural Language Inference) feladatra: minden minta egy premise-hipotézis
pár, és a feladat eldönteni, hogy a hipotézis következik-e a premise-ből
(entailment), ellentmond-e (contradiction), vagy semmilyen logikai
viszonyban nem áll vele (neutral).

Domain-eloszlás (10 zsáner): kormányzati szövegek, fikció,
telefonbeszélgetések átirata, levelezés, kormányzati jelentések,
rádió-műsorszövegek, populáris tudományos cikkek, szépirodalom-kritika,
9/11 jelentés, törvényszéki dokumentumok.

Ez nem hír-cikk-dataset. Az osztályozási teljesítmény ezért **természetszerűleg
gyengébb**, mint egy hír-domain-en fine-tuneolt modellé. A monitoring
küszöbök (`baseline_v1.json`) ezt a realitást tükrözik (55% accuracy
baseline, nem 92%).

A bart-large-mnli model card részletes leírást ad a training-eljárásról:
[huggingface.co/facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli).

## 2. Kiértékelő dataset: AG News stratifikált sample

Forrás: [`ag_news`](https://huggingface.co/datasets/ag_news) test split
(7600 cikk, 4 kategória)

A teljes AG News test split-ből egy 100 cikkes sample-t emelünk ki
**stratifikálva** (`scripts/build_ag_news_eval.py`):

- **Kategória × hossz-bin**: 4 kategória (`World`, `Sports`, `Business`,
  `Sci/Tech`) × 3 token-hossz-bin (`short` < 40 tok, `medium` 40-69 tok,
  `long` >= 70 tok) × 8 cikk = 96 cikk + 4 véletlen kiegészítés.
- **Reprodukálhatóság**: `random_seed=42` rögzítve a build szkriptben.
- **Verziózott artifakt**: `data/eval/ag_news_eval_v1.csv` (28 KB) +
  `data/eval/ag_news_eval_v1.meta.json` (a build paramétereinek snapshot-je).

A sample mérete (100) nem statisztikai erő-szempontból optimális, hanem
a **lefutási idő** korlát (CPU-n a teljes sample 2-3 perc, MPS-en
~30 másodperc) és **Git-méret-szempont** (28 KB elfér Git-ben, 7600 cikk
~1.5 MB-tal nem).

A meta-fájl a stratifikált bin-eloszlást is rögzíti:

```
length_bin     count
short            34
medium           34
long             32
```

### Miért stratifikáltunk hossz szerint?

Egy egyszerű random sample esetén a hossz-eloszlás visszaadná az AG News
természetes eloszlását (medián 53 token), ami ~normál körüli. A
stratifikálás biztosítja, hogy a `short` és `long` bin is reprezentált
legyen, így a `notebook/evaluation.ipynb` per-bin accuracy-t tud számolni.
A baseline-mérés szerint a hossz-bin szerinti accuracy:

```
length_bin    accuracy
short         61.8%
medium        44.1%
long          59.4%
```

Ez érdekes mintázat — a medium gyengébb, mint a long. Lehet random
ingadozás (kis sample), vagy a medium binbe kerültek olyan ambiguus
politikai/üzleti cikkek, ahol a kategorizálás szubjektív.

## 3. Truncation-edge-case dataset: Long articles

Forrás: [`SetFit/bbc-news`](https://huggingface.co/datasets/SetFit/bbc-news)
(1000 cikk, 5 kategória) + 4 kézzel írt health cikk

Az AG News-ban a leghosszabb cikk is csak 132 token, így a 1024-es
modell-limit ott soha nem aktiválódik. A truncation-viselkedés
tesztelésére (`data/eval/long_articles_eval_v1.csv`, 16 cikk):

- **BBC News** 4 kategóriájából (`sport`, `business`, `politics`, `tech`)
  3-3 cikk, csak 400+ tokenes filter-rel (`scripts/build_long_articles_eval.py`).
  A BBC kategóriák a `labels_v1.json` mi-formátumunkra képezve.
- **4 kézzel írt health cikk**: a BBC News-ban nincs health kategória,
  és a publikus health-dataset-ek tipikusan strukturáltak (PubMed
  abstract, klinikai jelentés). A kézi cikkeket WHO/CDC közleményekből
  inspirálva állítottuk össze: kardiovaszkuláris egészség, antibiotikum-
  rezisztencia, mRNS-vakcinák, tinédzser-mentális egészség.

Token-eloszlás:
```
true_label              count  min  mean   max
business                    3  401   538   803
health                      4  212   226   243
politics                    3  498   567   696
science and technology      3  462   764  1075
sports                      3  403   554   850
```

A `science and technology` kategóriában van egy **1075 tokenes cikk**, ami
átlépi a 900-tokenes truncation-küszöböt. A baseline-méréskor pontosan ezt
a cikket a modell rosszul kategorizálta (sci/tech → world news), ami
konkrét bizonyíték a truncation információ-veszteségére.

## Címke-set: `labels_v1.json`

A modellnek átadott candidate labels listája egy verziózott JSON fájl
(`data/prompts/labels_v1.json`):

```json
{
  "version": "v1",
  "model": "facebook/bart-large-mnli",
  "hypothesis_template": "This text is about {}.",
  "labels": [
    "world news",
    "sports",
    "business",
    "science and technology",
    "politics",
    "health"
  ],
  "label_aliases": {
    "World": "world news",
    "Sports": "sports",
    "Business": "business",
    "Sci/Tech": "science and technology"
  }
}
```

A 6 címke az AG News 4 alap-kategóriájából + 2 további (`politics`, `health`)
áll össze. Ez tudatos választás: bemutatja a zero-shot rugalmasságát —
új kategória felvétele mindössze a JSON szerkesztését igényli, retraining
nincs.

A `label_aliases` mező az AG News eredeti címkéit (pl. `Sci/Tech`) képezi
le a mi long-form formátumunkra (pl. `science and technology`). A
hosszabb forma jobb NLI-hipotézist eredményez ("This text is about
science and technology" természetesebb, mint "This text is about
Sci/Tech."), és a baseline-méréskor ~1-3 százalékponttal jobb accuracy-t
ad.

## Verziózási stratégia röviden

| Artifakt | Verzió | Mikor új verzió? |
|---|---|---|
| `labels_v1.json` | v1 | Címke változás, hipotézis-template változás, modell-csere |
| `ag_news_eval_v1.csv` | v1 | Új eval-szabály (más bin-határok), nagyobb sample |
| `long_articles_eval_v1.csv` | v1 | Új kategóriák, új cikkek hozzáadása |
| `baseline_v1.json` | v1 | Az evaluation.ipynb újrafuttatása új modellverzióra |

Egyik fájl sem mutálódik soha — új verzió minden változás. A régi
verziók maradnak, így bármely korábbi kiértékelés rekonstruálható.

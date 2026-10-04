# Dairy Farm Animal ID Recovery — ANSC 6060 Mini Project

> Recovering missing animal identifiers from automated milking system records using biological fingerprinting, Wood's lactation curves, and globally optimal assignment.

---

## Table of Contents
- [Project Overview](#project-overview)
- [The Problem](#the-problem)
- [Repository Structure](#repository-structure)
- [Naming Conventions](#naming-conventions)
- [Environment & Tools](#environment--tools)
- [Data Lineage](#data-lineage)
- [Data Cleaning Strategy](#data-cleaning-strategy)
- [Key Discoveries](#key-discoveries)
- [Modelling Strategy](#modelling-strategy)
- [All Models Tried](#all-models-tried)
- [Final Methodology](#final-methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Farm Recommendations](#farm-recommendations)

---

## Project Overview

Automated Milking Systems (AMS) on dairy farms record detailed per-session milking data for every cow. Each record is identified by an **AnimalId** — a unique hashed integer assigned to the cow's RFID ear tag. When a cow loses or damages her tag, the milking data is still captured by the sensors but the identifier is missing.

This project develops a multi-model pipeline to **recover the missing AnimalId** for approximately 1.7 million records (20% of the dataset). The approach combines biological lactation modelling, multi-gap identity fingerprinting, and the Hungarian optimal assignment algorithm to achieve record-level identification with zero conflicts.

---

## The Problem

| Fact | Value |
|---|---|
| Total records | 8,495,421 |
| Records with missing AnimalId | 1,701,003 (20.0%) |
| Unique known animals | 9,087 |
| Date range | 2019-06-14 to 2021-10-24 |
| Milking sessions per day | 3 (morning, midday, evening) |
| Gap animals confirmed | 5,617 (gaps >30 days) |
| Truly confirmed gap animals (>60d) | 4,253 |

### Why records go missing
In commercial dairy operations, cows are identified at the milking robot by an RFID tag. If the tag falls off, is damaged, or fails to scan, the system still records all sensor measurements but cannot attach an animal identifier — the **dropped RFID tag problem**.

### Why this is hard
On any given day, approximately **683 cows are missing simultaneously**. Their 2,049 milking records are pooled together with no timestamps, no stall IDs, and no milking order. The mean yield difference between competing animals is only **0.165 kg** — smaller than natural daily variation of ~1.1 kg. No single-day signal reliably separates them.

### Why it matters
Without AnimalId, records cannot be linked to lactation history, reproductive status, or days in milk — making them unusable for herd management and productivity analysis.

---

## Repository Structure

```
ANSC4040-AnimalIDRecovery/
│
├── notebooks/
│   └── DataAnalysis_Modeling.ipynb       # EDA, cleaning, all models
│
├── data/                                 # ← gitignored
│   └── Data_set_prep_assignment_1.csv    # original source (read-only)
│
├── outputs/                              # ← gitignored
│   ├── AnimalIDRecovered_Final.csv       # final recovered dataset
│   ├── DfKnown_clean.csv                 # cleaned known records
│   ├── DfMissing_clean.csv               # protected missing records
│   ├── DfGaps.csv                        # gap periods per animal
│   ├── AllWoodsParams.pkl                # fitted Wood's curve parameters
│   ├── AllProfiles.pkl                   # animal feature profiles
│   ├── WoodsPredDictAll.pkl              # pre-computed yield predictions
│   └── DfHungarian.csv                   # Hungarian assignment results
│
├── .gitignore
├── LICENSE
└── README.md
```

> **Data privacy:** The dataset contains hashed animal identifiers and commercial farm data. Raw and processed data files are excluded via `.gitignore`.

---

## Naming Conventions

### Python variables
PascalCase throughout

| Pattern | Example | Used for |
|---|---|---|
| `Df` prefix | `DfKnown`, `DfMissing`, `DfGaps` | DataFrames |
| `Dct` prefix | `WoodsPredDictAll`, `AnimalsByDate` | Dictionaries |
| `Clf` prefix | `ClfSGD`, `ClfRF` | Classifiers |
| Descriptive suffix | `ClusterFeatures`, `FilteredAnimals` | Lists and configs |
| No underscores in columns | `AnimalIdRecovered`, `ExpectedDIM` | DataFrame columns |

### Files
- Notebooks: `PascalCase.ipynb`
- Outputs: descriptive names, no spaces
- Checkpoints: saved after every heavy computation to avoid reruns

---

## Environment & Tools

| Component | Choice |
|---|---|
| IDE | Google Colab (cloud) |
| Storage | Google Drive (`/content/drive/MyDrive/MiniProject`) |
| Language | Python 3.13 |
| Key libraries | pandas, numpy, scikit-learn, scipy |
| Key functions | `scipy.optimize.curve_fit`, `scipy.optimize.linear_sum_assignment` |
| Version control | GitHub (public repo, data excluded) |

---

## Data Lineage

```
Raw CSV — Data_set_prep_assignment_1.csv (8,495,421 rows × 11 columns)
    │  READ ONLY — never modified
    │
    ▼
DataAnalysis_Modeling.ipynb
    │
    ├── Load & rename columns
    ├── Parse EventDate to datetime
    │
    ├── Rulebook & Metadata → MiniProjectRuleBook&MetaData.xlsx
    │     Domain rules, valid ranges, outlier methods per column
    │
    ├── Data Quality Audit
    │     Missing %, negatives, zeros, invalid sessions, date range
    │
    ├── Missingness Pattern Analysis
    │     AnimalId, LactationNumber, DaysInMilk, ReproductionStatus
    │     always missing together (block missingness confirmed)
    │     Mean yield known = missing = 13.93 kg (random tag failure)
    │
    ├── Outlier Flagging (IQR + domain rules)
    │     16 flag columns added — used as features, not for dropping
    │
    ├── Data Cleaning (known rows only — missing rows never touched)
    │     Dropped: AvgFlowIsZero (215), DurationOver600s (194),
    │              AverageMilkFlowKgPerMin missing (172)
    │     Removed exact duplicates
    │
    ├── DfKnown (6,737,364 rows) — saved to Drive
    ├── DfMissing (1,701,003 rows) — saved to Drive, never modified
    │
    ├── Gap Analysis
    │     420,670 total gaps found; 8,930 gaps >30 days
    │     5,617 animals with meaningful gaps
    │     96.8% of gaps contain a calving event
    │     CalvingDate estimated as PostDate − PostDIM
    │
    ├── Biological Profiling
    │     Wood's curve fitted per animal per lactation
    │     Expected yield projected for every gap day
    │     597,822 predictions built
    │
    ├── Hungarian Optimal Assignment (per date per session)
    │     Biological pre-filter → ~225 plausible animals per date
    │     Cost matrix built → solved in 0.041s
    │     Zero conflicts guaranteed
    │
    ├── Confidence filtering
    │     HIGH: |ActYield − ExpYield| < 0.5 kg → 586,311 records
    │
    └── Output → AnimalIDRecovered_Final.csv (8,438,367 rows)
```

---

## Data Cleaning Strategy

### Dropped rows (known data only — missing rows were not dropped)

| Reason | Count |
|---|---|
| AverageMilkFlowKgPerMin = 0 (sensor error) | 215 |
| MilkingDurationSeconds > 600s (logging error) | 194 |
| AverageMilkFlowKgPerMin missing | 172 |
| Exact duplicates (same animal, date, session, measurements) | Variable per run |

### Kept as model features
- `DaysInMilkOver350` — late lactation, real biology
- `DurationUnder150s` — fast milking, correlates with late lactation
- `TotalMilkYieldSessionKgIQRHigh` — high producers
- `Flow30To60IsZero` — possible slow let-down

### Key finding
Missingness is a clean block — all 4 identifier columns are always missing together. Mean yield of known and missing records is identical (13.93 kg)

---

## Key Discoveries

| Discovery | Finding |
|---|---|
| Ghost herd structure | ~2,101 rotating animals × 270 days × 3 sessions = 1,701,003 exactly, and 683 random animals missing eash day  |
| Tag loss prevalence | 62% of known animals (5,584) lost their tag at least once |
| Gap animals confirmed | 4,253 animals disappeared then reappeared with known identity |
| Calving during gap | 96.8% (4,115/4,253) calved during their gap; 99.8% dates estimable |
| Cluster purity | Only 5% — records from ~3.4 different animals always mixed per cluster |
| Unique signatures | 87.3% of multi-gap biological signatures are unique to one animal |
| Wood's curve accuracy | 0.014 kg mean prediction error across 66 gap days |
| Yield separability | Only 3 of 697 animals have a unique yield range on any given date |
| Natural daily variation | ~1.1 kg std within one animal at a fixed DIM window |
| Competitor yield gap | Mean difference between competing animals: 0.165 kg < variation |

---

## Modelling Strategy

### Why simple classifiers failed
Individual milking records are not distinctive enough to identify one cow among thousands with similar production levels. The mean yield difference between competing animals (0.165 kg) is smaller than natural daily variation (1.1 kg). No single-day, single-feature signal reliably separates them.

### The correct framing
The problem is not "which known animal does this record look like?" It is "given that Animal_101 was absent from date A to date B, which mystery records across that entire period belong to them?" — a gap-period-level matching problem, not a record-level classification problem.

### Why greedy assignment failed
Greedy approaches (each animal independently picks its best record) cause conflicts where 3.4 animals compete for the same record. The best trajectory always wins, leaving others empty. 94.7% of gap days have a valid match available but greedy assignment only achieves 24% coverage.

### Why Hungarian works
The Hungarian algorithm finds the globally optimal one-to-one assignment across all animals simultaneously. No animal steals another's best match. Zero conflicts are mathematically guaranteed. A 225×560 cost matrix is solved in 0.041 seconds.

---

## All Models Tried

| Model | Result | Status |
|---|---|---|
| KMeans (683 clusters) | Wrong structure — assumed fixed ghost animals | ❌ Failed |
| KMeans (2,101 clusters) | 5% cluster purity — 3.4 animals mixed per cluster | ❌ Failed |
| DTW trajectory matching | 0.1% simulation accuracy — clusters too mixed | ❌ Failed |
| KNN on individual records | 0.87% — records indistinguishable individually | ❌ Failed |
| KNN on animal fingerprints | 1.0% top-1 — test animal always removed from pool | ❌ Failed |
| Wood's curve standalone | 0.7% top-1 — multi-lactation fitting problem | ❌ Failed |
| SGD linear classifier | 0.13% — features don't separate linearly | ❌ Failed |
| Random Forest (50 animals) | 36.0% (18× random) — proves features discriminative | ⚠️ Can't scale |
| Random Forest (4,253 animals) | 2.9% (123× random) — can't run at scale | ⚠️ Can't scale |
| Longitudinal cluster matching | 858,748 records · 99.4% cluster-level validation | ⚠️ Each cluster contains ~3.4 animals mixed — record-level assignment unvalidated |
| Multi-gap biological fingerprinting | 94.8% animal-level accuracy | ✅ Works |
| Wood's curve trajectory | 0.014 kg error · 94.7% day coverage | ✅ Works |
| Greedy trajectory assignment | 411,806 records · 285,675 conflicts | ⚠️ Conflicts |
| **Hungarian optimal assignment** | **586,311 records · 0 conflicts** | ✅ Final |

---

## Final Methodology

### Step 1 — Identify gap animals
Find all 5,617 known animals with gaps >30 days. Build exact absence date ranges from DfKnown chronology. Estimate calving dates as `PostDate − PostDIM days`.

### Step 2 — Build biological fingerprints
For each animal: gap duration(s), lactation number at entry/exit, DIM at entry/exit, yield at entry/exit. 87.3% of combined multi-gap signatures are unique to one animal (94.8% animal-level identification accuracy on simulation).

### Step 3 — Fit Wood's lactation curve
```
y = A × t^B × e^(-Ct)
```
Fitted per animal per lactation using `scipy.optimize.curve_fit`. 5,610 of 5,617 animals fitted successfully. Mean prediction error: 0.014 kg.

### Step 4 — Project expected yield for every gap day
For each gap date: calculate exact expected DIM (increases by 1/day, resets at estimated calving date), look up Wood's curve prediction at that DIM. 597,822 predictions built across all animals and gap dates.

### Step 5 — Biological pre-filtering (per date per session)
Include only animals whose expected yield > 3 kg (milking, not dry) AND whose expected yield is within ±1.5 std of at least one available mystery record. Reduces ~686 absent animals to ~225 plausible milking candidates per date-session.

### Step 6 — Build cost matrix
```python
Cost[animal, record] = 10 × |YieldDiff| / YieldStd
                     + |FlowDiff|  / FlowStd
                     + |DurDiff|   / DurStd
```
Yield weighted 10× as primary biological signal. Matrix size ~225 × 560.

### Step 7 — Hungarian algorithm
```python
from scipy.optimize import linear_sum_assignment
RowIdx, ColIdx = linear_sum_assignment(CostMatrix)
```
Globally optimal one-to-one assignment. Solved in 0.041 seconds per date-session. Run across all 827 dates × 3 sessions in ~190 seconds total.

### Step 8 — Confidence filtering
Keep only HIGH confidence: `|ActYield − ExpYield| < 0.5 kg` (within natural daily variation).

### Step 9 — Iterative enrichment
HIGH confidence recovered records added back to each animal's known pool. Wood's curves refitted with enriched data (mean 127 records added per animal across 4,628 animals). Second round of Hungarian run with improved predictions.

### Features used
- `TotalMilkYieldSessionKg`
- `MilkingDurationSeconds`
- `AverageMilkFlowKgPerMin`
- `MilkFlow30To60SecondsKgPerMin` (profile only)
- `MilkYieldFirst2MinutesKg` (profile only)

All features standardised with `StandardScaler` before cost matrix computation.

---

## Results

### Recovery summary

| Method | Records | Percentage | Confidence |
|---|---|---|---|
| Hungarian HIGH (YieldDiff < 0.5 kg) | 586,311 | 34.5% | HIGH |
| Unrecovered | 1,114,692 | 65.5% | — |
| **Total missing** | **1,701,003** | **100%** | — |

### Assignment quality

| Metric | Value |
|---|---|
| Total HIGH confidence records | 586,311 |
| Animals re-identified | 4,628 |
| Conflicts (records assigned to 2+ animals) | 0 |
| Mean yield difference | 0.124 kg |
| Within ±0.5 kg of Wood's prediction | 84.1% |
| Within ±1.0 kg | 88.2% |
| Animal-level identification accuracy | 94.8% |

### Sanity checks

| Check | Expected | Result | Status |
|---|---|---|---|
| Ghost herd count | 2,101 × 270 × 3 = 1,701,003 | 1,701,003 | ✓ |
| Yield bias | Known mean = Missing mean | 13.93 kg = 13.93 kg | ✓ |
| Conflicts | 0 | 0 | ✓ |
| Calving dates within gap | >99% | 4,106/4,115 (99.8%) | ✓ |
| Unique animal signatures | High % | 87.3% | ✓ |
| Recovered IDs in DfKnown | 100% | 4,628/4,628 | ✓ |

### Output file: `AnimalIDRecovered_Final.csv`

| RecoveryMethod | Records | Description |
|---|---|---|
| Original | 6,737,364 | Known records — tag always scanned |
| Recovered | 586,311 | AnimalId recovered — HIGH confidence |
| Unrecovered | 1,114,692 | Cannot identify without timestamps |
| **Total** | **8,438,367** | |

### Yield distribution (known vs recovered)

| Statistic | Known | Recovered |
|---|---|---|
| Mean | 13.93 kg | 13.93 kg |
| Std | 3.64 kg | 3.61 kg |
| Median | 13.74 kg | 13.71 kg |

---

## Limitations

- **65.5% unrecoverable** — 1,114,692 records belong to animals with no matching biological profile in DfKnown, or to known gap animals on days where natural yield variation pushed the true record outside the ±0.5 kg HIGH confidence threshold.
- **Cluster impurity** — earlier cluster-based approaches showed only 5% purity (3.4 animals per cluster). Individual record assignment is fundamentally limited by overlapping yield ranges.
- **Yield separability** — only 3 of 697 animals have a unique yield range on any given date. No algorithm can separate the remaining 694 without additional sensor data.
- **No ground truth** — the true AnimalId for missing records is unknown by definition. Validation relies on simulation (removing known animals and re-identifying them) and biological consistency checks.
- **Wood's curve limitation** — the curve assumes a single smooth lactation trajectory. High-DIM or multi-lactation periods may have prediction error larger than 0.5 kg, causing valid records to be excluded from HIGH confidence.

---

## Farm Recommendations

| Priority | Action | Impact |
|---|---|---|
| Critical | Add milking **timestamps** per session | Enables 100% future recovery — links S1/S2/S3 to same cow |
| Critical | Add **stall or reader ID** | Physically separates animals at each milking point |
| High | Audit **tag attachment method** | 62% tag loss rate is critically high — consider double-tagging |
| Medium | Daily alert when missing count > 100 | Early detection of tag loss events |
| Medium | **Veterinary review** of gap animals | 4,115 animals calved during tag-loss; reproductive records may be incomplete |
| Low | Double-tag high-value animals | Backup identification for highest-producing cows |

---

## License

MIT License — see `LICENSE` file.

---

*ANSC 6060 Mini Project · Google Colab · 2026*

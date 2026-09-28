# Dairy Farm Animal ID Recovery — ANSC 6060 Mini Project

> Recovering missing animal identifiers from automated milking system records using unsupervised machine learning.

---

## Table of Contents
- [Project Overview](#project-overview)
- [The Problem](#the-problem)
- [Repository Structure](#repository-structure)
- [Naming Conventions](#naming-conventions)
- [Environment & Tools](#environment--tools)
- [Data Lineage](#data-lineage)
- [Data Cleaning Strategy](#data-cleaning-strategy)
- [Modelling Strategy](#modelling-strategy)
- [Results](#results)
- [Limitations](#limitations)
- [Timeline](#timeline)

---

## Project Overview

Automated Milking Systems (AMS) on dairy farms record detailed per-session milking data for every cow. Each record is identified by an **AnimalId** — a unique hashed integer assigned to the cow's RFID ear tag. When a cow loses or damages her RFID tag, the milking data is still captured by the sensors but the identifier is missing. This project develops a machine learning pipeline to **recover the missing AnimalId** for approximately 1.7 million records (20% of the dataset) using the milking feature data that was successfully recorded.

---

## The Problem

| Fact | Value |
|---|---|
| Total records | 8,495,421 |
| Records with missing AnimalId | 1,701,003 (20.0%) |
| Unique known animals | 9,087 |
| Date range | 2019-06-14 to 2021-10-24 |
| Milking sessions per day | 3 (probably: morning, midday, evening) |
| Estimated ghost animals | 683 |

### Why records go missing
In commercial dairy operations, cows are identified at the milking robot by an RFID tag attached to their ear or leg. If the tag falls off, is damaged, or fails to scan, the milking system still records all sensor measurements (yield, flow rate, duration) but cannot attach an animal identifier. This is the **dropped RFID tag problem**.

Analysis revealed the missingness is **not random** — approximately 683 animals appear on every single date with no identifier recorded, a stable 20% of the active herd across all 830 unique dates.

### Why this matters
Without AnimalId, records cannot be linked to lactation history, reproductive status, or days in milk — making them unusable for herd management decisions and productivity analysis.

---

## Repository Structure

```
ANSC4040-AnimalIDRecovery/
│
├── notebooks/
│   └── DataAnalysis_Modeling.ipynb    # single notebook: EDA, cleaning, modelling
│
├── data/                              # ← gitignored, not shared publicly
│   └── Data_set_prep_assignment_1.csv # original source (read-only in code)
│
├── outputs/                           
│   ├── AnimalIDRecovered.csv          # final recovered dataset, ← gitignored
│   └── MiniProjectRuleBook&MetaData.xlsx
│
├── .gitignore
├── LICENSE
└── README.md
```

> **Data privacy:** The dataset contains hashed animal identifiers and commercial farm data. Raw and processed data files are excluded from this repository via `.gitignore`.

---

## Naming Conventions

### Python variables
PascalCase throughout — no snake_case, no ALL_CAPS constants.

| Pattern | Example | Used for |
|---|---|---|
| `Df` prefix | `DfKnown`, `DfMissing`, `DfMaster` | DataFrames |
| Descriptive suffix | `ClusterFeatures`, `MatchCols` | Lists and configs |
| `Scaler` prefix | `ScalerCluster` | Sklearn scalers |
| `Nn` prefix | `NnInitial`, `NnExtended` | NearestNeighbors objects |
| `X` prefix | `XMissingScaled`, `XKnownProfiles` | Feature matrices |
| No underscores in column names | `AnimalIdRecovered`, `LactationNumberFilled` | DataFrame columns |

### Files
- Notebooks: `PascalCase.ipynb`
- Outputs: descriptive names, no spaces — `AnimalIDRecovered.csv`
- Excel: `MiniProjectRuleBook&MetaData.xlsx`

---

## Environment & Tools

| Component | Choice |
|---|---|
| IDE | Google Colab (cloud, no local setup) |
| Storage | Google Drive (`/content/drive/MyDrive/MiniProject`) |
| Language | Python 3.13 |
| Key libraries | pandas, numpy, scikit-learn, ydata-profiling |
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
    ├── Load & rename (6 columns renamed for clarity)
    ├── Parse EventDate to datetime
    │
    ├── Rulebook & Metadata → MiniProjectRuleBook&MetaData.xlsx
    │     Domain rules, valid ranges, outlier methods per column
    │
    ├── Data Quality Audit
    │     Missing %, negatives, zeros, invalid sessions, date range
    │
    ├── Missingness Pattern Analysis
    │     Confirmed: AnimalId, LactationNumber, DaysInMilk,
    │     ReproductionStatus always missing together (block missingness)
    │
    ├── Outlier Flagging (IQR + domain rules)
    │     16 flag columns added — used as features, not for dropping
    │
    ├── Data Cleaning (known rows only — missing rows never touched)
    │     Dropped: AvgFlowIsZero (215), DurationOver600s (194),
    │              AverageMilkFlowKgPerMin missing (172)
    │     Removed duplicates (same animal, date, session, measurements)
    │
    ├── Phase 1 — Exact Match
    │     Key: EventDate + MilkingSession + TotalMilkYieldSessionKg
    │     Resolved: 47,781 records (2.8%)
    │
    ├── Ghost Animal Discovery
    │     683 × 3 × 830 = 1,700,670 ≈ 1,701,003 actual missing (100% match)
    │     Same 683 animals missing every day — systematic scanner failure
    │
    ├── Phase 2 — MiniBatchKMeans Clustering
    │     Clustered 1,700,962 missing records into 683 clusters
    │     Mean cluster size: 2,490 records (= 830 days × 3 sessions ✓)
    │     Matched each centroid to nearest known animal profile
    │     Resolved: 1,653,181 records (97.2%)
    │
    ├── Phase 3 — Fill Identifier Columns
    │     LactationNumber, DaysInMilk, ReproductionStatus filled
    │     via nearest-date merge_asof per recovered animal
    │
    └── Output → AnimalIDRecovered.csv (8,438,367 rows)
```

---

## Data Cleaning Strategy

### Dropped rows (known data only — missing rows never dropped)

| Reason | Count |
|---|---|
| AverageMilkFlowKgPerMin = 0 (sensor error) | 215 |
| MilkingDurationSeconds > 600s (logging error) | 194 |
| AverageMilkFlowKgPerMin missing | 172 |
| Exact duplicates (same animal, date, session, measurements) | TBD on rerun |

### Kept as model features (not dropped)
- `DaysInMilkOver350` — late lactation cows, real biology
- `DurationUnder150s` — fast milking, correlates with late lactation
- `TotalMilkYieldSessionKgIQRHigh` — high producers
- `Flow30To60IsZero` — possible slow let-down

### Key finding
Missingness is a clean block — all 4 identifier columns (`AnimalId`, `LactationNumber`, `DaysInMilk`, `ReproductionStatus`) are always missing together. Only 4 partial-missing rows exist in 8.5M records.

---

## Modelling Strategy

### Why not KNN?
Initial attempts using K-Nearest Neighbours (global and date-windowed) achieved ~2% accuracy. The feature distributions of known and missing rows are nearly identical — individual milking records are not distinctive enough to identify one cow among thousands with similar production levels.

### The breakthrough: ghost animal discovery
Mathematical analysis revealed that exactly 683 ghost animals × 3 sessions × 830 days = 1,700,670 ≈ 1,701,003 actual missing records (100% match). The same fixed group of 683 animals appears on every date with no identifier. This changed the problem from classification to clustering.

### Pipeline

**Phase 1 — Exact Match (deterministic)**
Match missing rows to known animals on `EventDate + MilkingSession + TotalMilkYieldSessionKg`. Only assign where the match is unique. Resolved 47,781 records (2.8%) with 100% confidence.

**Phase 2 — MiniBatchKMeans Clustering**
- Cluster all 1,700,962 missing records (with complete features) into exactly 683 clusters
- Each cluster represents one ghost animal's complete milking history
- Mean cluster size: 2,490 records = 830 days × 3 sessions ✓
- Match each cluster centroid to the nearest known animal profile using Euclidean distance on 5 scaled milking features
- Greedy deduplication with 50 nearest neighbours ensures unique animal assignment per cluster
- 41 rows missing `AverageMilkFlowKgPerMin` assigned via median imputation + nearest centroid

**Phase 3 — Fill identifier columns**
Once `AnimalId` is recovered, fill `LactationNumber`, `DaysInMilk`, and `ReproductionStatus` by nearest-date lookup from the known records of the assigned animal using `pd.merge_asof`.

### Features used for clustering
- `TotalMilkYieldSessionKg`
- `MilkingDurationSeconds`
- `AverageMilkFlowKgPerMin`
- `MilkFlow30To60SecondsKgPerMin`
- `MilkYieldFirst2MinutesKg`

All features standardised with `StandardScaler` before clustering and distance computation.

---

## Results

### Recovery summary

| Method | Records | Percentage |
|---|---|---|
| Exact Match | 47,781 | 2.8% |
| Clustering | 1,653,181 | 97.2% |
| Clustering (imputed flow) | 41 | 0.0% |
| **Total recovered** | **1,700,836** | **99.99%** |
| Remaining null | 167 | 0.01% |

### Clustering validation

| Metric | Value |
|---|---|
| Target clusters | 683 |
| Clusters created | 683 |
| Mean cluster size | 2,490 records |
| Expected (830×3) | 2,490 records ✓ |
| Median match distance | 0.163 (low = centroids close to known profiles) |
| Unique animals assigned | 682 / 683 |

### Sanity checks

| Check | Result |
|---|---|
| Null identifiers in recovered | 167 (0.01%) |
| Recovered IDs outside known set | 0 ✓ |
| Session distribution preserved | ✓ |
| Invalid LactationNumber | 167 (same 167 null rows) |
| Invalid ReproductionStatus | 167 (same 167 null rows) |
| Yield distribution match (mean) | Known 13.93 vs Recovered 13.93 ✓ |

### Yield distribution (known vs recovered)

| Statistic | Known | Recovered |
|---|---|---|
| Mean | 13.93 kg | 13.93 kg |
| Std | 3.64 kg | 3.65 kg |
| Min | 5.03 kg | 5.03 kg |
| Median | 13.74 kg | 13.74 kg |
| Max | 54.84 kg | 53.98 kg |

---

## Limitations

- **682 of 683 unique animal assignments** — one cluster could not be uniquely resolved within the top 50 nearest profiles, resulting in one animal assigned to two clusters.
- **167 unfilled identifier rows** — rows where the recovered animal has no known records close enough in date for `merge_asof` to fill from. These remain null in the final dataset.
- **No ground truth** — because the true AnimalId for missing rows is genuinely unknown, the recovery accuracy cannot be directly verified. Validation relies on mathematical consistency (cluster size = 830×3), yield distribution matching, and domain sanity checks.
- **Unique animals in recovered: 682 not 683** — the scattering of 6,602 unique IDs seen in sanity check 2 reflects that the dedup greedy assignment reached 682 unique animals rather than 683.

---

## Timeline

| Week | Task | Status |
|---|---|---|
| Week 1 | Data inspection, metadata, profiling report | ✅ Complete |
| Week 2 | Data cleaning, outlier flagging, rulebook, EDA | ✅ Complete |
| Week 3 | Exact match, KNN baseline, ghost animal discovery, clustering | ✅ Complete |
| Week 4 | Poster, final notebook clean-up, GitHub submission | 🔲 In progress |

---

## License

MIT License — see `LICENSE` file.

---

*ANSC 4040 Mini Project · Google Colab · 2024*

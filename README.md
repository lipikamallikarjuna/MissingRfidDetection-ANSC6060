# Dairy Farm Animal ID Recovery — ANSC 6060 Mini Project

> Recovering missing animal identifiers from automated milking system records using machine learning.

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
- [Train / Validation / Test Split](#train--validation--test-split)
- [Timeline](#timeline)
- [Results & Findings](#results--findings)

---

## Project Overview

Automated Milking Systems (AMS) on dairy farms record detailed per-session milking data for every cow. Each record is identified by an **AnimalId** — a unique hashed integer assigned to the cow's RFID ear tag. When a cow loses or damages her RFID tag, the milking data is still captured by the sensors but the identifier is missing. This project develops a machine learning pipeline to **recover the missing AnimalId** for approximately 1.7 million records (20% of the dataset) using the milking feature data that was successfully recorded.

---

## The Problem

| Fact | Value |
|---|---|
| Total records | 8,495,421 |
| Records with missing AnimalId | 1,701,003 (20.02%) |
| Unique known animals | 9,087 |
| Date range | 2019-06-14 to 2021-10-24 |
| Milking sessions per day | 3 (probably morning, midday, evening) |

### Why records go missing
In commercial dairy operations, cows are identified at the milking robot by an RFID tag attached to their ear or leg. If the tag falls off, is damaged, or fails to scan, the milking system still records all sensor measurements (yield, flow rate, duration) but cannot attach an animal identifier. Approximately 700 animals per day are affected consistently throughout the dataset.

### Why this matters
Without AnimalId, records cannot be linked to:
- Lactation history (`LactationNumber`)
- Reproductive status (`ReproductionStatus`)
- Days in milk (`DaysInMilk`)

Recovering the identifier restores the full record and enables downstream analysis of animal health, productivity trends, and herd management decisions.

---

## Repository Structure

```
ANSC4040-MiniProject/
│
├── data/                          # ← gitignored (see .gitignore)
│   └── Data_set_prep_assignment_1.csv
│
├── notebooks/
│   ├── 01_MetaDataRulebook.ipynb      # metadata + rulebook creation
│   ├── 02_DataCleaning.ipynb          # cleaning, flagging, EDA
│   └── 03_AnimalID_Recovery.ipynb     # modelling pipeline
│
├── outputs/
│   ├── MiniProjectRulebook.xlsx       # domain rulebook (gitignored)
│   └── DairyFarmData_ProfileReport.html
│
├── .gitignore
├── LICENSE
└── README.md                          # ← this file
```

---

## Naming Conventions

### Files
- Notebooks prefixed with two-digit index: `01_`, `02_`, `03_`
- Snake_case for all filenames: `animal_id_recovery.ipynb`
- Outputs include PascalCase descriptor: `MiniProjectRulebook.xlsx`

### Variables (Python)
| Pattern | Example | Used for |
|---|---|---|
| `df_` prefix | `df_known`, `df_missing`, `df_master` | DataFrames |
| `UPPER_CASE` | `FEATURES`, `WINDOW_DAYS`, `K` | Constants |
| `_path` suffix | `data_file_path`, `output_file_path` | File paths |
| `n_` prefix | `n_dupes`, `n_total` | Counts |
| `_flag` suffix | `AvgFlow_IsZero`, `Duration_Over600s` | Boolean flag columns |
| `_KNN` suffix | `AnimalId_KNN` | Model-predicted columns |
| `_Filled` suffix | `LactationNumber_Filled` | Imputed columns |
| `_Recovered` suffix | `AnimalId_Recovered` | Final recovered columns |

### Column naming
- PascalCase with unit appended: `TotalMilkYieldSessionKg`, `MilkingDurationSeconds`
- Original names renamed on load for clarity (see notebook 01)

### Git branches
- `main` — stable, submitted version
- `dev` — active development
- `feature/model-name` — experimental model branches

---

## Environment & Tools

| Component | Choice | Reason |
|---|---|---|
| IDE | Google Colab | Cloud GPU/CPU, no local setup, shareable |
| Storage | Google Drive (`/content/drive/MyDrive/MiniProject`) | Persistent across Colab sessions |
| Language | Python 3.13 | Standard for data science |
| Key libraries | pandas, numpy, scikit-learn, ydata-profiling, matplotlib | Industry standard |
| Version control | GitHub (public repo) | Required by assignment |

### Why Colab over local
- Dataset is 8.5M rows — Colab's 12GB RAM handles it without memory issues on a local machine
- No environment setup required across different machines
- Easy to share and reproduce results

---

## Data Lineage

```
Raw CSV (8,495,421 rows × 11 columns)
    │
    ▼
01_MetaDataRulebook.ipynb (done)
    │  • Column renaming (6 columns)
    │  • EventDate parsed to datetime
    │  • Metadata table generated
    │  • Domain rulebook created (ValidMin, ValidMax, OutlierMethod, flags)
    │  • Saved: MiniProjectRulebook.xlsx
    │
    ▼
02_DataCleaning.ipynb (done)
    │  • Data quality audit (missing, zeros, negatives, invalid)
    │  • Missingness pattern confirmed: block-missing (all 4 ID columns together)
    │  • Outlier flagging: IQR fences + domain hard rules (16 flag columns)
    │  • Rows dropped from known only (581 rows):
    │      - AvgFlow_IsZero (215 rows)
    │      - Duration_Over600s (194 rows)  
    │      - AverageMilkFlowKgPerMin missing (172 rows)
    │  • Duplicate removal (known rows only)
    │  • Split: df_known (6,794,418) + df_missing (1,701,003)
    │  • Saved: Data_set_prep_assignment_1.csv (overwrite)
    │
    ▼
03_AnimalID_Recovery.ipynb (in-process, methodology may change)
    │  • Phase 1: Exact match (47,781 resolved, 2.8%)
    │  • Phase 2: Longitudinal profile KNN (1,653,222 rows)
    │  • Phase 3: Fill LactationNumber, DaysInMilk, ReproductionStatus
    │  • Saved: final dataset (overwrite)
    ▼
Final dataset: 0 missing AnimalId rows
```

---

## Data Cleaning Strategy

### What was cleaned
| Issue | Action | Rows affected |
|---|---|---|
| Zero average milk flow | Dropped (known rows only) | 215 |
| Session duration >600s | Dropped (known rows only) | 194 |
| AverageMilkFlowKgPerMin missing | Dropped (known rows only) | 172 |
| Exact duplicate records | Dropped (keep first) | TBD |
| Missing AnimalId block | **Protected — never dropped** | 1,701,003 |

### What was flagged but kept (model features)
- `DaysInMilk_Over350` — late lactation cows, real biology
- `Duration_Under150s` — fast milking, correlates with late lactation
- `TotalMilkYieldSessionKg_IQR_High` — high producers
- `MilkingDurationSeconds_IQR_High` — long sessions
- `Flow3060_IsZero` — possible slow let-down, kept pending investigation

### Key finding during cleaning
Missingness is **not random**. Approximately 700 animals per day consistently have no identifier recorded — a stable 20% of the active herd on any given date. This pattern persisted across all 830 unique dates in the dataset, pointing to a systematic RFID scanner failure rather than individual tag loss events.

---

## Modelling Strategy (methodology may change)

### Approach evolution
Initial attempts at single-record KNN (global and date-windowed) achieved only ~2% validation accuracy. Investigation revealed that individual milking records are not distinctive enough to identify cows — two cows with similar production levels look identical in a single session.

The solution is **longitudinal profile matching**: a cow's rolling average yield over 7 days has a coefficient of variation of only 8.9%, making it a stable fingerprint when compared against the same animal's known profile.

### Phase 1 — Exact Match (deterministic)
- Match key: `EventDate + MilkingSession + TotalMilkYieldSessionKg`
- Only assign where the match is unique (one candidate animal)
- Result: 47,781 records resolved (2.8%)
- Confidence: 100% — no model uncertainty

### Phase 2 — Longitudinal Profile KNN
- Build 7-day rolling mean profile per animal per feature
- Features: `TotalMilkYieldSessionKg`, `MilkingDurationSeconds`, `AverageMilkFlowKgPerMin`, `MilkFlow30To60SecondsKgPerMin`, `MilkYieldFirst2MinutesKg`
- Date window: ±15 days (candidates must have been recently active)
- Standardise features with `StandardScaler` before distance computation
- K=5 (tuned across K=1,3,5,10,15)
- Confidence score: proportion of K neighbours agreeing
- Output: `AnimalId_KNN` + `KNN_Confidence`

### Why KNN
- No assumption about distribution of milking features
- Naturally handles the multi-class problem (9,087 classes)
- Interpretable: nearest neighbours can be inspected
- Longitudinal profiles reduce within-animal variance (CV 8.9%) making distances meaningful

### Phase 3 — Fill remaining identifier columns
Once `AnimalId` is recovered, look up that animal's known record closest in `EventDate` and fill:
- `LactationNumber` — from nearest known record
- `DaysInMilk` — from nearest known record
- `ReproductionStatus` — modal value within ±30 days

---

## Train / Validation / Test Split

| Split | Size | Purpose |
|---|---|---|
| Train | 90% of df_known | Build KNN index |
| Validation | 10% of df_known (capped 30K) | Tune K, report accuracy |
| "Test" | df_missing (1.7M rows) | Final prediction — no ground truth |

**Note:** A traditional held-out test set with ground truth does not exist for the missing rows — by definition their true AnimalId is unknown. The validation set on known rows is therefore the primary performance metric. Validation is stratified by animal (10% per animal) to ensure all 9,087 animals are represented in training.

---

## Timeline

| Week | Task | Status |
|---|---|---|
| Week 1 | Data inspection, metadata table, profiling report |  Complete |
| Week 2 | Data cleaning, outlier flagging, rulebook, EDA |  Complete |
| Week 3 | Exact match, KNN baseline, longitudinal profiling |  In progress |
| Week 4 | Final model, fill remaining columns, export, write-up, poster |  Upcoming |

### Detailed Week 3–4 plan
- [x] Exact match phase (47,781 resolved)
- [x] Global KNN baseline (2% accuracy — documented as finding)
- [x] Date-windowed KNN (2% accuracy — documented as finding)
- [x] CV analysis → confirmed yield is stable fingerprint (CV 8.9%)
- [ ] Longitudinal profile KNN validation
- [ ] K tuning
- [ ] Apply to 1.7M missing rows
- [ ] Fill LactationNumber, DaysInMilk, ReproductionStatus
- [ ] Export final dataset
- [ ] Create Poster

---

## Results & Findings

| Finding | Detail |
|---|---|
| Missingness type | Systematic, not random — same ~700 animals missing daily |
| Cause | Likely RFID tag scanner failure for a fixed group of animals |
| Single-record KNN accuracy | ~2% — features not distinctive at record level |
| Yield CV (7-day rolling) | 8.9% — yield IS a stable longitudinal fingerprint |
| Exact match resolution | 47,781 records (2.8%) resolved with 100% confidence |
| Profile KNN accuracy | TBD — in progress |

---

## License

MIT License — see `LICENSE` file.

---

*Project by: Lipika Mallikarjuna | ANSC 4040 | Google Colab | 2026*

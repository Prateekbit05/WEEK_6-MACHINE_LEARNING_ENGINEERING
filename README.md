# Week 6 — Day 1: Data Pipeline + EDA + Project Architecture

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Project Structure](#project-structure)
- [Day 1 Implementation](#day-1-implementation)
  - [Data Pipeline](#data-pipeline)
  - [EDA Notebook](#eda-notebook)
  - [Processed Dataset](#processed-dataset)
  - [Data Report](#data-report)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Learnings](#key-learnings)
- [Deliverables Checklist](#deliverables-checklist)
- [Author](#author)

---

## Overview

This project demonstrates **production-grade ML data pipeline architecture** using real-world network intrusion detection data (NF-UQ-NIDS-v2). Day 1 focuses on building a clean, modular data ingestion and preprocessing pipeline followed by comprehensive Exploratory Data Analysis.

**Objective:** Design a professional ML project structure, load and clean raw data, generate EDA insights, and produce a versioned processed dataset ready for feature engineering in Day 2.

---

## Learning Objectives

By completing Day 1, I gained hands-on experience with:

- Professional ML project folder architecture
- Train/validation/test splitting strategies
- Handling missing values (mean/median/interpolation)
- Outlier detection using Z-score and IQR methods
- Data scaling with StandardScaler and MinMaxScaler
- Class imbalance detection (SMOTE / class weights)
- Dataset versioning via folder hashing and metadata JSON
- EDA reporting: correlation matrices, distributions, missing value heatmaps

---

## Project Structure

```
DAY_1-DATA_PIPELINE_EDA/
│
├── diagnose_dataset.py
├── run_day1.sh
├── requirements.txt
├── README.md
│
├── docs/
│   ├── COMMANDS.md
│   ├── README.md
│   ├── THEORY.md
│   └── TOPICS-INFO.md
│
├── logs/
│   └── 20260215_pipeline.log
│
└── src/
    ├── config/
    │   └── config.yaml
    ├── data/
    │   ├── dataset_info.json
    │   ├── external/
    │   ├── metadata/
    │   │   └── version_15d6bde0.json
    │   ├── processed/
    │   │   ├── feature_info.json
    │   │   ├── final.csv
    │   │   ├── pipeline_results.json
    │   │   ├── sample.csv
    │   │   ├── test.csv
    │   │   └── train.csv
    │   └── raw/
    │       └── NF-UQ-NIDS-v2.csv
    ├── notebooks/
    │   ├── EDA.ipynb
    │   ├── generate_eda_large.py
    │   └── plots/
    │       ├── correlation.png
    │       ├── numerical_distributions.png
    │       └── target_distribution.png
    ├── pipelines/
    │   ├── data_pipeline.py
    │   └── __init__.py
    ├── reports/
    │   ├── DATA-REPORT.md
    │   ├── boxplots_outliers.png
    │   ├── correlation_matrix_full.png
    │   ├── correlation_matrix_top15.png
    │   ├── feature_by_target.png
    │   ├── high_correlations.csv
    │   ├── missing_values.png
    │   ├── numerical_statistics.csv
    │   ├── outlier_analysis.csv
    │   ├── outlier_percentage.png
    │   └── target_distribution.png
    └── utils/
        ├── helpers.py
        ├── logger.py
        └── __init__.py
```

---

## Day 1 Implementation

### Data Pipeline

**Topics Covered:**
- Config-driven pipeline using `config.yaml`
- Missing value handling (mean/median imputation)
- Duplicate row removal
- Outlier detection using Z-score and IQR methods
- Feature scaling with StandardScaler / MinMaxScaler
- Class imbalance logging and detection
- Dataset versioning via metadata JSON hashing

**Exercise:**
- Loads raw `NF-UQ-NIDS-v2.csv` from `/data/raw/`
- Cleans data: missing values, duplicates, outliers
- Saves cleaned splits to `/data/processed/`: `final.csv`, `train.csv`, `test.csv`
- Logs every pipeline stage with timestamps to `/logs/`

**Deliverables:**
- `data_pipeline.py` — Modular pipeline with cleaning, scaling, and versioning
- `config.yaml` — Configurable paths, thresholds, and scaler settings

---

### EDA Notebook

**Topics Covered:**
- Correlation matrix heatmaps (top 15 and full feature set)
- Numerical feature distribution plots
- Target class distribution analysis (attack vs. benign traffic)
- Missing values heatmap for data quality assessment
- Boxplots for per-feature outlier visualization
- Statistical summary: mean, std, skewness, kurtosis

**Exercise:**
- Runs full EDA on the processed dataset
- Generates and saves all plots to `/notebooks/plots/` and `/reports/`
- Identifies top correlated features with the target variable

**Deliverables:**
- `EDA.ipynb` — Interactive EDA notebook with all analysis and plots

#### Sample Plots

> Correlation matrix, target distribution, and outlier boxplots available in `src/reports/`

---

### Processed Dataset

**Topics Covered:**
- Stratified train/test splitting
- Processed feature storage with metadata
- Pipeline results logging to JSON

**Exercise:**
- Derived from raw `NF-UQ-NIDS-v2.csv` after full pipeline execution
- Missing values imputed, duplicates removed, outliers treated
- Features scaled and class distribution documented in `feature_info.json`

**Deliverables:**
- `final.csv` — Full cleaned dataset
- `train.csv` / `test.csv` — Stratified splits for model training
- `sample.csv` — Small sample for quick debugging
- `pipeline_results.json` — Pipeline execution summary

---

### Data Report

**Topics Covered:**
- Dataset overview documentation
- Preprocessing decision logging
- Outlier and correlation findings summary

**Exercise:**
- Documents dataset shape, feature types, and target distribution
- Summarizes missing value counts and imputation strategies applied
- Records outlier analysis results with Z-score and IQR thresholds
- Lists top correlated features with the target variable

**Deliverables:**
- `DATA-REPORT.md` — Full findings report covering all pipeline decisions and EDA results

---

## Prerequisites

Before running Day 1, ensure you have:

- **Python:** Version 3.10 or higher
- **pip:** For installing dependencies
- **Jupyter:** For running EDA notebook
- **Git:** For version control

### Installation Commands

```bash
# Clone the repo
git clone git@github.com:Prateekbit05/WEEK_6-MACHINE_LEARNING_ENGINEERING.git
cd WEEK_6-MACHINE_LEARNING_ENGINEERING/DAY_1-DATA_PIPELINE_EDA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import pandas, sklearn, seaborn, matplotlib; print('All packages OK')"
```

---

## Quick Start

### Run the Full Pipeline

```bash
cd DAY_1-DATA_PIPELINE_EDA

# Activate virtual environment
source venv/bin/activate

# Run the complete pipeline (data loading → cleaning → EDA → reports)
bash run_day1.sh

# Or run pipeline directly
python src/pipelines/data_pipeline.py
```

### Run EDA Notebook

```bash
# Launch Jupyter
jupyter notebook src/notebooks/EDA.ipynb
```

### Diagnose Dataset

```bash
# Run dataset diagnostics (shape, nulls, dtypes)
python diagnose_dataset.py
```

### Check Pipeline Logs

```bash
# View latest pipeline log
cat src/logs/$(ls src/logs/ | tail -1)
```

### Output Locations

```
Processed data  →  src/data/processed/final.csv
EDA plots       →  src/notebooks/plots/
Full reports    →  src/reports/
Pipeline logs   →  src/logs/
Data report     →  src/reports/DATA-REPORT.md
```

---

## Key Learnings

### ML Project Architecture
- Why a clean, modular folder structure matters in production ML
- Separating raw vs. processed data to maintain reproducibility
- Config-driven pipelines to avoid hardcoded values

### Data Preprocessing
- Mean/median imputation strategies and when to use each
- Z-score vs. IQR for outlier detection: tradeoffs and use cases
- StandardScaler vs. MinMaxScaler based on data distribution

### Dataset Versioning
- Hashing processed datasets for reproducibility
- Logging pipeline metadata (shape, features, timestamp) to JSON
- Why versioning matters when models are retrained over time

### EDA Best Practices
- Correlation matrices to identify multicollinearity before modeling
- Class imbalance detection and its impact on model evaluation
- Outlier percentage analysis to make informed preprocessing decisions

### Production Habits
- Timestamped log files for every pipeline run
- Separating utility functions into `utils/` for reuse across days
- Documenting every data decision in `DATA-REPORT.md`

---

## Deliverables Checklist

| Deliverable | File | Status |
|---|---|---|
| Data Pipeline | `src/pipelines/data_pipeline.py` | ✅ |
| EDA Notebook | `src/notebooks/EDA.ipynb` | ✅ |
| Processed Dataset | `src/data/processed/final.csv` | ✅ |
| Data Report | `src/reports/DATA-REPORT.md` | ✅ |
| Documentation | `docs/` | ✅ |
| Pipeline Logs | `src/logs/` | ✅ |

---

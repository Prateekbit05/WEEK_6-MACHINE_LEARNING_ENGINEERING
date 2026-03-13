# Week 6: Machine Learning Engineering
# (Data Science + Feature Engineering + Model Building + Deployment + Monitoring)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Project Structure](#project-structure)
- [Day-wise Implementation](#day-wise-implementation)
  - [Day 1: Data Pipeline + EDA + Project Architecture](#day-1-data-pipeline--eda--project-architecture)
  - [Day 2: Feature Engineering + Feature Selection](#day-2-feature-engineering--feature-selection)
  - [Day 3: Model Building + Advanced Training Pipeline](#day-3-model-building--advanced-training-pipeline)
  - [Day 4: Hyperparameter Tuning + Explainability + Error Analysis](#day-4-hyperparameter-tuning--explainability--error-analysis)
  - [Day 5: Model Deployment + Monitoring + MLOps (Capstone)](#day-5-model-deployment--monitoring--mlops-capstone)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Learnings](#key-learnings)
- [Author](#author)

---

## Overview

This project demonstrates a **production-grade Machine Learning Engineering pipeline** built on the **NF-UQ-NIDS-v2 Network Intrusion Detection dataset**. Over 5 days, I built a complete end-to-end ML system — from raw data ingestion and feature engineering, through multi-model training and hyperparameter tuning, to a fully containerised FastAPI deployment with real-time drift monitoring.

**Objective:** Understand how production ML systems are designed, trained, optimised, deployed, and monitored — covering the full lifecycle from raw data to a running API.

---

## Learning Objectives

By completing this week's tasks, I gained hands-on experience with:

- Professional ML pipeline architecture and modular system design
- Exploratory Data Analysis (EDA) and dataset versioning
- Advanced feature engineering, transformation, and ensemble feature selection
- Multi-model training with cross-validation and overfitting control
- Hyperparameter tuning using GridSearch, RandomSearch, and Bayesian Optimization (Optuna)
- Model explainability using SHAP values and feature importance
- Model deployment with FastAPI — REST endpoints, input validation, request tracking
- Data drift detection using PSI and KS tests
- Docker containerisation and production-ready service configuration

---

## Project Structure

```
WEEK_6-MACHINE_LEARNING_ENGINEERING/
│
├── DAY-1-DATA-PIPELINE/
│   ├── src/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── processed/
│   │   │   └── metadata/
│   │   ├── pipelines/
│   │   │   └── data_pipeline.py
│   │   ├── notebooks/
│   │   │   └── EDA.ipynb
│   │   ├── utils/
│   │   ├── config/
│   │   └── logs/
│   ├── DATA-REPORT.md
│   └── README.md
│
├── DAY-2-FEATURE-ENGINEERING/
│   ├── src/
│   │   ├── features/
│   │   │   ├── build_features.py
│   │   │   ├── feature_selector.py
│   │   │   ├── transformers.py
│   │   │   ├── feature_list.json
│   │   │   └── engineered_features.json
│   │   └── data/processed/
│   ├── FEATURE-ENGINEERING-DOC.md
│   └── README.md
│
├── DAY-3-MODEL-BUILDING/
│   ├── src/
│   │   ├── training/
│   │   │   ├── train.py
│   │   │   ├── models.py
│   │   │   └── cv_trainer.py
│   │   ├── models/
│   │   │   └── best_model.pkl
│   │   └── evaluation/
│   │       └── metrics.json
│   ├── plots/
│   ├── reports/
│   │   └── MODEL-COMPARISON.md
│   └── README.md
│
├── DAY-4-HYPERPARAMETER-TUNING/
│   ├── src/
│   │   ├── training/
│   │   │   └── tuning.py
│   │   └── evaluation/
│   │       └── shap_analysis.py
│   ├── outputs/
│   │   ├── models/
│   │   └── plots/
│   ├── tuning/
│   │   └── results.json
│   ├── MODEL-INTERPRETATION.md
│   └── README.md
│
├── DAY-5-DEPLOYMENT/
│   ├── deployment/
│   │   ├── api.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── schemas.py
│   │   └── Dockerfile
│   ├── monitoring/
│   │   ├── drift_checker.py
│   │   └── accuracy_monitor.py
│   ├── models/
│   │   └── v1/
│   ├── logs/
│   │   ├── prediction_logs.csv
│   │   └── drift_report_<timestamp>.json
│   ├── dashboard/
│   │   └── dashboard.py
│   ├── docker-compose.yml
│   ├── .env.example
│   ├── requirements.txt
│   ├── DEPLOYMENT-NOTES.md
│   └── README.md
│
└── README.md
```

---

## Day-wise Implementation

### Day 1: Data Pipeline + EDA + Project Architecture

**Topics Covered:**
- ML project architecture and professional folder structure
- Train/validation/test splitting strategy
- Handling missing values (mean/median/interpolate)
- Outlier detection using Z-score and IQR
- Data scaling with StandardScaler and MinMaxScaler
- Class imbalance handling with SMOTE
- Dataset versioning via dataframe hashing

**Exercise:**
Built a production-ready data loader and EDA pipeline that:
- Loads the NF-UQ-NIDS-v2 dataset from `/data/raw` using chunked reading (50k rows/chunk)
- Cleans data — handles missing values, duplicates, and outliers
- Exports `/data/processed/final.csv` — cleaned, stratified splits ready for training
- Generates a full EDA report: correlation matrix, feature distributions, target class balance, and missing value heatmap

**Dataset Summary:**

| Property | Value |
|---|---|
| Dataset | NF-UQ-NIDS-v2 (Network Flow) |
| Working Sample | 100,000 rows |
| Total Features | 46 columns (42 numerical, 4 categorical) |
| Target Column | `Attack` |
| Number of Classes | 20 attack types |
| Class Imbalance Ratio | 32,986 : 1 |

**Deliverables:**
- `pipelines/data_pipeline.py` — DataPipeline class with chunked loading
- `notebooks/EDA.ipynb` — Full exploratory analysis notebook
- `data/processed/final.csv` — Cleaned dataset
- `DATA-REPORT.md` — EDA findings and recommendations

---

### Day 2: Feature Engineering + Feature Selection

**Topics Covered:**
- Categorical encoding (OneHot, Target, Label)
- Numerical feature transformations (log, sqrt, power)
- Interaction and ratio feature generation
- Binning and aggregation features
- Feature selection: Correlation threshold, Mutual Information, RFE, Tree-based importance

**Exercise:**
Built a full feature engineering pipeline that:
- Transformed the cleaned 42-column dataset into a rich **144-feature** engineered dataset
- Applied 8 transformer types — log, sqrt, power, interaction, ratio, aggregation, binning, statistical
- Reduced to the **50 most informative features** using a 6-method ensemble voting selector
- Produced `X_train_final.csv` and `X_test_final.csv` — zero data leakage, scaler-deferred

**Pipeline Results:**

| Stage | Features | Action |
|---|---|---|
| Input from Day 1 | 42 | Cleaned numerical features |
| After Engineering | 144 | +102 new features created |
| After Variance Filter | — | Removed near-zero variance |
| After Correlation Filter | — | Removed corr > 0.95 pairs |
| **Final Output** | **50** | Ensemble voting selection |

**Deliverables:**
- `features/build_features.py` — FeatureEngineer class, 8 transformer types
- `features/feature_selector.py` — FeatureSelector class, 6 selection methods
- `features/feature_list.json` — 50 selected features with full importance scores
- `FEATURE-ENGINEERING-DOC.md` — Summary documentation

---

### Day 3: Model Building + Advanced Training Pipeline

**Topics Covered:**
- Multi-model training: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network
- 5-fold cross-validation
- Overfitting vs underfitting control
- Regularization (L1/L2)
- Model comparison and automated best model selection

**Exercise:**
Created a unified training pipeline that:
- Trained 5 models with 5-fold cross-validation
- Wrapped each model in a `StandardScaler → Model` sklearn Pipeline to prevent leakage
- Output accuracy, precision, recall, F1, and ROC-AUC for all models
- Automatically saved the best model (by CV F1) to `models/best_model.pkl`

**Results Summary:**

| Model | CV Accuracy | CV F1 | Train Time |
|-------|-------------|-------|------------|
| Logistic Regression | 0.0073 | 0.0105 | 158.2s |
| Random Forest | 0.1420 | 0.1784 | 17.4s |
| Neural Network | 0.3285 | 0.1779 | 51.1s |
| XGBoost | 0.3190 | 0.2262 | 81.5s |
| **LightGBM ✅** | **0.3172** | **0.2343** | 94.7s |

**Deliverables:**
- `training/train.py` — Main training pipeline
- `models/best_model.pkl` — Auto-saved best model
- `evaluation/metrics.json` — All CV and test metrics
- `reports/MODEL-COMPARISON.md` — Auto-generated comparison report

---

### Day 4: Hyperparameter Tuning + Explainability + Error Analysis

**Topics Covered:**
- Hyperparameter tuning: GridSearch, RandomSearch, Bayesian Optimization (Optuna TPE)
- SHAP values and TreeExplainer
- Feature importance ranking
- Bias/variance analysis
- Error analysis and overfitting diagnosis

**Exercise:**
Implemented a full tuning and explainability pipeline that:
- Tuned Random Forest and XGBoost using GridSearch, RandomSearch, or Optuna (30 trials)
- Ran Bayesian optimization with Optuna TPE sampler for smarter search
- Generated SHAP beeswarm summary plot and feature importance bar chart
- Exported ranked feature importance to CSV for downstream analysis

**Tuning Results (Optuna — 30 trials, Random Forest):**

| | Accuracy | F1 Score (Macro) |
|--|----------|-----------------|
| **Baseline** | 0.9888 | 0.9873 |
| **Tuned** | 0.9867 | 0.9850 |
| **Δ Change** | -0.0021 | -0.0024 |

**Best Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 292 |
| `max_depth` | 20 |
| `min_samples_split` | 8 |
| `min_samples_leaf` | 2 |

**Deliverables:**
- `training/tuning.py` — HyperparameterTuner class with 3 search strategies
- `evaluation/shap_analysis.py` — SHAPAnalyzer class with TreeExplainer
- `tuning/results.json` — Best params and model comparison
- `MODEL-INTERPRETATION.md` — Explainability documentation

---

### Day 5: Model Deployment + Monitoring + MLOps (Capstone)

**Topics Covered:**
- Serving models using FastAPI with uvicorn
- Creating `/predict` endpoint with Pydantic input validation
- Saving and loading models with joblib
- Prediction logging and request ID tracking with UUID
- Data drift detection: PSI (Population Stability Index) + KS test
- Concept drift via Chi-square on label distributions
- Docker containerisation with multi-stage builds

**Exercise (Capstone):**
Deployed the trained NF-UQ-NIDS-v2 classifier as a production-grade API with:
- A `POST /predict` endpoint with full Pydantic schema validation
- UUID-based request ID tracking per prediction
- Append-only prediction logging to `prediction_logs.csv`
- Versioned model loading from `models/v1/`
- Batch prediction endpoint `/predict/batch`
- Hot model reload via `POST /model/reload`
- P95 latency tracking with a bounded `deque(maxlen=1000)`
- Drift checker running independently against prediction logs using PSI + KS + Chi-square

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info + status |
| `GET` | `/health` | Health check + uptime |
| `GET` | `/metrics` | Request counts, P95 latency, drift warnings |
| `GET` | `/model/info` | Model version, feature count, class map |
| `POST` | `/predict` | Single prediction with live drift check |
| `POST` | `/predict/batch` | Batch predictions |
| `POST` | `/model/reload` | Hot-reload model from disk |

**Monitoring Coverage:**

| Type | Method | Threshold | Alert |
|------|--------|-----------|-------|
| Covariate drift | PSI per feature | PSI > 0.2 | `overall_drift: true` |
| Distribution shift | KS test | p < 0.05 | `ks_drift: true` |
| Concept drift | Chi-square on labels | p < 0.05 | `concept_drift: true` |
| Anomaly spike | Benign % drop | < 56% traffic | `anomaly_flag: true` |
| Per-request drift | Z-score (live) | z > threshold | `drift_warning: true` |

**Deliverables:**
- `deployment/api.py` — FastAPI app with all endpoints
- `deployment/Dockerfile` — Multi-stage production Docker image
- `monitoring/drift_checker.py` — PSI + KS + Chi-square drift detection
- `logs/prediction_logs.csv` — Append-only prediction log
- `DEPLOYMENT-NOTES.md` — Architecture and deployment documentation

---

## 🛠️ Prerequisites

Before running this project, ensure you have:

- **Python:** Version 3.10 or higher
- **Docker:** Version 20.10 or higher (for Day 5)
- **Docker Compose:** Version 2.0 or higher (for Day 5)
- **Jupyter Notebook:** For running EDA notebooks (Day 1)
- **Git:** For version control

### Installation Commands:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Docker (Ubuntu/Debian)
sudo apt update
sudo apt install docker.io docker-compose

# Verify installations
python --version
docker --version
docker-compose --version
jupyter --version
```

---

## Quick Start

### Day 1: Data Pipeline + EDA

```bash
cd DAY-1-DATA-PIPELINE

# Install dependencies
pip install -r requirements.txt

# Place raw dataset
cp your-dataset.csv src/data/raw/data.csv

# Run full pipeline
python src/pipelines/data_pipeline.py

# Run EDA notebook
jupyter notebook src/notebooks/EDA.ipynb
```

### Day 2: Feature Engineering

```bash
cd DAY-2-FEATURE-ENGINEERING

# Step 1 — Engineer features (42 → 144)
python src/features/build_features.py

# Step 2 — Select best features (144 → 50)
python src/features/feature_selector.py

# Run both in sequence
python src/features/build_features.py && python src/features/feature_selector.py
```

### Day 3: Model Training

```bash
cd DAY-3-MODEL-BUILDING

# Run full training pipeline
python src/training/train.py

# Output files:
# src/models/best_model.pkl
# src/evaluation/metrics.json
# plots/model_comparison.png
# reports/MODEL-COMPARISON.md
```

### Day 4: Hyperparameter Tuning + Explainability

```bash
cd DAY-4-HYPERPARAMETER-TUNING

# Step 1 — Run hyperparameter tuning
python src/training/tuning.py

# Step 2 — Run SHAP explainability
python src/evaluation/shap_analysis.py

# Choose tuning method at runtime (edit config.yaml or pass arg)
# Options: bayesian (recommended), random_search, grid_search
```

### Day 5: Model Deployment (Local)

```bash
cd DAY-5-DEPLOYMENT

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env

# Start API server
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload

# API docs available at:
# http://localhost:8000/docs   ← Swagger UI
# http://localhost:8000/redoc  ← ReDoc
```

### Day 5: Model Deployment (Docker)

```bash
cd DAY-5-DEPLOYMENT

# Build Docker image
docker build -t nids-api:v1 .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name nids-api \
  nids-api:v1

# Check health
curl http://localhost:8000/health

# Run drift checker
python -m monitoring.drift_checker --mode watch --interval 60

# Stop container
docker stop nids-api && docker rm nids-api
```

### Day 5: Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_12": 1.2, "feature_3": 0.0}}'
```

**Example Response:**

```json
{
  "request_id": "a3f1c2d4-7b8e-4f91-a2c3-d4e5f6a7b8c9",
  "model_version": "v1",
  "prediction": 0,
  "label": "Benign",
  "confidence": 0.982143,
  "drift_warning": false,
  "latency_ms": 4.231,
  "timestamp": "2026-02-20T10:15:30.123Z"
}
```

---

## Key Learnings

### Data Engineering
- How to handle large datasets with chunked loading to avoid memory overflow
- Stratified train/test splits to preserve class proportions in imbalanced datasets
- IQR-based outlier capping to preserve data volume compared to row removal
- Lightweight dataset versioning using dataframe hashing (no DVC required)

### Feature Engineering
- Fitting all transformers on train data only and applying to test — eliminates data leakage
- Ensemble voting for feature selection (≥2 method agreement) is more robust than trusting a single selector
- Deferring scaling to model training so each model type (tree vs linear) applies its preferred scaler

### Model Building
- Wrapping models in `StandardScaler → Model` sklearn Pipelines prevents leakage during cross-validation
- Using CV F1 as the selection criterion is more reliable than accuracy for imbalanced multi-class problems
- `label_binarize` with OvR strategy enables ROC-AUC on 16-class classification

### Hyperparameter Tuning
- Optuna TPE sampler builds a probabilistic model of good regions — smarter and faster than grid search
- A small accuracy drop after tuning is normal — the tuned model generalises better than an overfit baseline
- SHAP `TreeExplainer` computes exact (not approximate) Shapley values for tree-based models

### Model Deployment
- `ModelState` singleton pattern keeps all counters, model refs, and latency history in one in-memory object
- `deque(maxlen=1000)` provides bounded memory for latency tracking — never grows unbounded in production
- Multi-stage Docker builds produce smaller, safer final images by separating build and runtime environments
- `asynccontextmanager` lifespan replaces deprecated `on_startup`/`on_shutdown` in modern FastAPI

### Monitoring
- PSI catches gradual distribution shifts; KS test catches sudden ones — using both reduces false negatives
- Dual-layer drift checking: z-score per request in `/predict` (real-time) + PSI/KS in `drift_checker.py` (batch)
- Welford's online algorithm streams large CSVs in chunks for computing training stats without RAM overflow

---

## Useful Commands

```bash
# Data Pipeline
python src/pipelines/data_pipeline.py          # Run full data pipeline
jupyter notebook src/notebooks/EDA.ipynb       # Launch EDA notebook

# Feature Engineering
python src/features/build_features.py          # Engineer features
python src/features/feature_selector.py        # Select top features

# Model Training
python src/training/train.py                   # Train all models + save best

# Tuning & Explainability
python src/training/tuning.py                  # Run hyperparameter tuning
python src/evaluation/shap_analysis.py         # Run SHAP analysis

# API (Local)
uvicorn deployment.api:app --reload            # Start dev server
curl http://localhost:8000/health              # Health check
curl http://localhost:8000/metrics             # View metrics

# Drift Monitoring
python -m monitoring.drift_checker --mode report         # One-time drift report
python -m monitoring.drift_checker --mode watch          # Continuous watch loop
python -m monitoring.drift_checker --mode accuracy       # Concept drift check

# Docker
docker build -t nids-api:v1 .                  # Build image
docker run -d -p 8000:8000 nids-api:v1         # Run container
docker-compose up -d                            # Start all services
docker-compose logs -f                          # Follow logs
docker-compose down                             # Stop all services
```

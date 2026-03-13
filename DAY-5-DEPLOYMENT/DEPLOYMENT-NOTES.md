# DEPLOYMENT-NOTES.md — Day 5: Model Deployment + Monitoring + MLOps Capstone
## NF-UQ-NIDS-v2 Network Intrusion Detection System

**Generated:** 2026-03-13
**Model Version:** v1
**Framework:** FastAPI + Streamlit + Docker

---

## 1. Project Summary

This document covers the complete deployment pipeline for the NIDS ML system. The system serves a trained Random Forest classifier (from Day 4) via a production-grade FastAPI server with drift monitoring, prediction logging, and a live Streamlit dashboard.

| Week 6 Stage | Output |
|---|---|
| Day 1 — Data Pipeline | `final.csv`, `DATA-REPORT.md` |
| Day 2 — Feature Engineering | `X_train_final.csv` (20 features), `feature_list.json` |
| Day 3 — Model Training | `best_model.pkl`, `metrics.json` |
| Day 4 — Tuning + Explainability | `model.joblib`, `scaler.joblib`, SHAP plots |
| Day 5 — Deployment + Monitoring | FastAPI, Drift Checker, Dashboard |

---

## 2. Folder Structure

```
DAY-5-DEPLOYMENT/
├── deployment/
│   ├── api.py              # FastAPI app — all endpoints
│   ├── schemas.py          # Pydantic request/response models
│   ├── model_loader.py     # Versioned model loading
│   ├── config.py           # Centralized config (paths, features, labels)
│   └── logger.py           # Logging setup
├── monitoring/
│   ├── drift_checker.py    # KS-test + PSI drift detection
│   └── accuracy_monitor.py # Confidence decay + latency + attack surge
├── dashboard/
│   └── dashboard.py        # Streamlit monitoring dashboard
├── models/
│   └── v1/
│       ├── model.joblib        # Trained Random Forest
│       ├── scaler.joblib       # Feature scaler
│       ├── feature_stats.json  # Drift baseline statistics
│       └── metrics.json        # Baseline evaluation metrics
├── logs/
│   ├── api.log                     # API server logs
│   ├── drift_checker.log           # Drift monitor logs
│   ├── prediction_logs.csv         # Per-request prediction log
│   └── drift_report_*.json         # Timestamped drift reports
├── inputs/
│   └── X_train_final.csv       # Training data for drift baseline
├── prediction_logs.csv         # Root-level prediction log
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── test_api.py
```

---

## 3. Deployed Model

| Property | Value |
|---|---|
| Model Type | Random Forest Classifier |
| Optimization | Optuna (Bayesian tuning) |
| Class Imbalance | SMOTE applied |
| Accuracy | 0.9867 |
| F1 Score (Macro) | 0.9850 |
| Features | 20 (feature_0 to feature_19) |
| Serialization | joblib |
| Model Path | `models/v1/model.joblib` |
| Scaler Path | `models/v1/scaler.joblib` |

### Label Map

| Class Index | Label |
|---|---|
| 0 | Benign |
| 1 | DoS/DDoS |
| 2 | Port Scan |
| 3 | Brute Force |
| 4 | Web Attack |
| 5 | Botnet |
| 6 | Infiltration |
| 7 | Heartbleed |
| 8 | Infiltration-Portscan |

---

## 4. FastAPI Application (`deployment/api.py`)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root — service name, version, status |
| GET | `/health` | Model loaded status, uptime, total requests |
| GET | `/metrics` | Live performance metrics (latency, drift count) |
| GET | `/model/info` | Model type, version, feature count, label map |
| POST | `/predict` | Single prediction with drift check and logging |
| POST | `/predict/batch` | Batch predictions (up to 1000 samples) |
| POST | `/model/reload` | Hot reload model from disk without restart |
| GET | `/docs` | Auto-generated Swagger UI |
| GET | `/redoc` | ReDoc API documentation |

### ModelState Singleton

The `ModelState` class manages all runtime state across requests:

| Property | Description |
|---|---|
| `model` | Loaded joblib Random Forest |
| `scaler` | Loaded joblib scaler |
| `feature_stats` | Loaded `feature_stats.json` for drift |
| `total_requests` | Lifetime request counter |
| `successful_predictions` | Successful prediction counter |
| `failed_predictions` | Error counter |
| `drift_warnings` | Total drift flags triggered |
| `latencies` | Rolling deque of last 1000 latencies |
| `avg_latency` | Mean of latency deque |
| `p95_latency` | 95th percentile latency |

### Prediction Flow

```
POST /predict
    │
    ├── 1. Validate input via PredictRequest (Pydantic)
    ├── 2. prepare_features() → align to FEATURE_COLS → scaler.transform()
    ├── 3. model.predict() → class index
    ├── 4. model.predict_proba() → confidence score
    ├── 5. check_drift() → Z-score per feature vs feature_stats.json
    ├── 6. log_prediction() → append row to prediction_logs.csv
    └── 7. Return PredictResponse
```

### Drift Detection Logic (`check_drift`)

- Compares each incoming feature against training baseline in `feature_stats.json`
- Computes Z-score: `|val - mean| / std` per feature
- If Z-score exceeds `DRIFT_ZSCORE_CUTOFF` (default: 5.0) → feature flagged
- If flagged features exceed `DRIFT_OUTLIER_RATIO` (default: 20%) → `drift_warning = True`
- Increments `state.drift_warnings` counter on each flagged prediction

---

## 5. Input / Output Schemas (`deployment/schemas.py`)

### POST /predict — Request

```json
{
  "features": {
    "feature_0": -0.26,
    "feature_1": 4.85,
    "feature_2": 5.35,
    "feature_3": 1.23,
    "feature_4": 2.25,
    "feature_5": -0.07,
    "feature_6": -2.03,
    "feature_7": 2.08,
    "feature_8": 1.20,
    "feature_9": -1.61,
    "feature_10": -1.52,
    "feature_11": 3.12,
    "feature_12": -0.61,
    "feature_13": 1.01,
    "feature_14": 2.00,
    "feature_15": -2.52,
    "feature_16": 0.70,
    "feature_17": -3.43,
    "feature_18": 1.64,
    "feature_19": -2.13
  }
}
```

### POST /predict — Response

```json
{
  "request_id": "a3f2c1d4-8e7b-4a0f-9b12-123456789abc",
  "model_version": "v1",
  "prediction": 0,
  "label": "Benign",
  "confidence": 0.943200,
  "drift_warning": false,
  "latency_ms": 12.453,
  "timestamp": "2026-03-13T07:50:45.123Z"
}
```

### Input Validation Rules

| Rule | Detail |
|---|---|
| Type enforcement | All feature values must be `int` or `float` |
| Null/NaN rejection | `np.isfinite()` check on every value |
| Batch size limit | 1 to 1000 samples per `/predict/batch` request |
| Missing features | Defaults to 0.0 for any missing `FEATURE_COLS` key |

---

## 6. Prediction Logging

Every `/predict` call appends one row to `prediction_logs.csv`:

| Column | Type | Description |
|---|---|---|
| `request_id` | UUID | Unique identifier per request |
| `timestamp` | datetime | UTC timestamp of prediction |
| `model_version` | string | Model version tag (e.g. `v1`) |
| `prediction` | int | Predicted class index |
| `label` | string | Human-readable class label |
| `confidence` | float | Max predict_proba score |
| `drift_warning` | bool | Z-score drift flag |
| `latency_ms` | float | End-to-end prediction latency |
| `features_json` | JSON string | Full input feature dictionary |

---

## 7. Monitoring

### Drift Checker (`monitoring/drift_checker.py`)

Two statistical methods are used in combination:

**KS-Test (Kolmogorov-Smirnov):**
- Simulates training distribution per feature using `feature_stats.json` (mean/std)
- Compares against production values from `prediction_logs.csv`
- Alert threshold: `p-value < 0.05`

**PSI (Population Stability Index):**
- Bins both distributions into 10 equal buckets
- Computes divergence score per feature

| PSI Range | Severity |
|---|---|
| PSI < 0.1 | LOW |
| 0.1 ≤ PSI < 0.2 | MEDIUM |
| PSI ≥ 0.2 | HIGH |

**Overall drift flag:** triggered when more than 20% of features show HIGH PSI or KS drift.

**Drift Report Output (`logs/drift_report_{timestamp}.json`):**

```json
{
  "report_time": "2026-03-13T07:50:45",
  "model_version": "v1",
  "samples_analyzed": 500,
  "features_checked": 20,
  "features_drifted": 2,
  "drift_fraction": 0.10,
  "overall_drift": false,
  "features": {
    "feature_0": {
      "psi": 0.045,
      "psi_severity": "LOW",
      "ks_stat": 0.032,
      "ks_p": 0.412,
      "ks_drift": false,
      "prod_mean": -0.241,
      "train_mean": -0.263
    }
  }
}
```

**Run Modes:**

| Mode | Command |
|---|---|
| Compute baseline | `python -m monitoring.drift_checker --mode compute-stats --dataset inputs/X_train_final.csv` |
| Drift report | `python -m monitoring.drift_checker --mode report` |
| Accuracy check | `python -m monitoring.drift_checker --mode accuracy` |
| Continuous watch | `python -m monitoring.drift_checker --mode watch --interval 60` |

---

### Accuracy Monitor (`monitoring/accuracy_monitor.py`)

Monitors prediction quality signals from `prediction_logs.csv` over a configurable time window.

| Alert | Trigger Condition |
|---|---|
| `confidence_drop` | Drop > 0.10 vs previous window |
| `latency_spike` | Current latency > 2× previous window |
| `attack_surge` | Attack traffic ratio exceeds 50% |
| `high_drift_rate` | More than 30% of predictions flagged for drift |

Report saved to: `logs/accuracy_monitor_{timestamp}.json`

```bash
# Single report
python -m monitoring.accuracy_monitor

# Continuous watch every 5 minutes
python -m monitoring.accuracy_monitor --watch --interval 300
```

---

## 8. Streamlit Dashboard (`dashboard/dashboard.py`)

```bash
streamlit run dashboard/dashboard.py
# Access at: http://localhost:8501
```

### Dashboard Sections

| Section | Description |
|---|---|
| API Status | Live health check — status, model version, uptime |
| Summary Metrics | Total predictions, window count, avg confidence, avg latency, drift flags |
| Prediction Distribution | Pie chart of label distribution |
| Predictions per Hour | Bar chart of hourly prediction volume |
| Hourly Avg Confidence | Line chart of confidence trend |
| Hourly Avg Latency | Line chart of latency trend |
| Log Explorer | Filterable table of last 200 predictions |
| Live Prediction Test | Input form to POST real `/predict` requests from browser |

### Sidebar Controls

| Control | Description |
|---|---|
| Auto Refresh | Refreshes page every 30 seconds via meta tag |
| Window (Hours) | Filter all metrics to last N hours (1–168) |
| API Docs link | Opens `/docs` Swagger UI |
| Health link | Opens `/health` JSON response |

---

## 9. Docker Deployment

### Build and Run

```bash
# Build image
docker build -t nids-ml-api:v1 .

# Run container
docker run -d \
  -p 8000:8000 \
  --name nids-api \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/prediction_logs.csv:/app/prediction_logs.csv \
  nids-ml-api:v1

# Verify
curl http://localhost:8000/health
```

### Docker Compose

```bash
docker-compose up -d       # Start all services
docker-compose ps          # View status
docker-compose logs -f     # Follow logs
docker-compose down        # Stop all
docker-compose down -v     # Stop and remove volumes
```

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | API bind host |
| `PORT` | `8000` | API port |
| `WORKERS` | `1` | Uvicorn worker count |
| `RELOAD` | `false` | Hot reload (dev only) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MODEL_VERSION` | `v1` | Active model version folder |
| `MODEL_FORMAT` | `joblib` | Serialization format |
| `DRIFT_ZSCORE_CUTOFF` | `5.0` | Z-score threshold per feature |
| `DRIFT_OUTLIER_RATIO` | `0.20` | Fraction of drifted features to trigger warning |
| `DRIFT_PSI_THRESHOLD` | `0.2` | PSI threshold for HIGH severity |
| `DRIFT_POLL_INTERVAL` | `300` | Watch mode poll interval (seconds) |

---

## 10. Output Files Reference

| File | Description |
|---|---|
| `deployment/api.py` | FastAPI app — all endpoints, ModelState, drift check, logging |
| `deployment/schemas.py` | Pydantic schemas for all request/response models |
| `deployment/config.py` | All paths, FEATURE_COLS, LABEL_MAP, env vars |
| `deployment/logger.py` | Timestamped file + console logging setup |
| `monitoring/drift_checker.py` | KS-test + PSI drift detection, 4 run modes |
| `monitoring/accuracy_monitor.py` | Confidence decay, latency spike, attack surge alerts |
| `dashboard/dashboard.py` | Streamlit live dashboard with prediction test UI |
| `models/v1/model.joblib` | Deployed Random Forest classifier |
| `models/v1/scaler.joblib` | Feature scaler for input preprocessing |
| `models/v1/feature_stats.json` | Per-feature mean/std/min/max for drift baseline |
| `models/v1/metrics.json` | Baseline evaluation metrics from Day 4 |
| `prediction_logs.csv` | Full prediction log with features_json per row |
| `logs/drift_report_*.json` | Timestamped KS + PSI drift check reports |
| `logs/accuracy_monitor_*.json` | Timestamped confidence + latency reports |
| `Dockerfile` | Python 3.12 slim container definition |
| `docker-compose.yml` | Full stack orchestration with volume mounts |
| `test_api.py` | API test script covering all endpoints |

---

## 11. Week 6 Engineering Conclusion

✔ 100,000 NIDS records cleaned and versioned — Day 1
✔ 102 new features engineered, 20 selected via multi-method voting — Day 2
✔ 5 models trained and compared with 5-fold cross-validation — Day 3
✔ Random Forest tuned with Optuna achieving 0.9867 accuracy — Day 4
✔ SHAP explainability with top-10 feature importance ranking — Day 4
✔ FastAPI deployed with 8 endpoints, Pydantic validation, UUID request tracking — Day 5
✔ KS-test + PSI drift detection across all 20 features — Day 5
✔ Confidence decay, latency spike, attack surge monitoring — Day 5
✔ Streamlit dashboard with live prediction test UI — Day 5
✔ Dockerized with docker-compose for production deployment — Day 5

---


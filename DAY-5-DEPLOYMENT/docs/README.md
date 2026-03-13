# 🚀 Day 5 — Model Deployment + Monitoring + MLOps (Capstone)

> **ML Engineering Week · Day 5 (Capstone)**
> Deploy the NF-UQ-NIDS-v2 classifier as a production-grade FastAPI service with drift monitoring, prediction logging, request ID tracking, and Docker containerisation.

---

## 📁 Project Structure

```
day-5/
├── deployment/
│   ├── api.py                  # 🔑 FastAPI app — all endpoints
│   ├── config.py               # Centralised config (cfg object)
│   ├── logger.py               # Structured logging setup
│   ├── schemas.py              # Pydantic request/response models
│   └── Dockerfile              # Multi-stage production Docker image
├── monitoring/
│   └── drift_checker.py        # 🔑 PSI + KS drift detection + watch loop
├── models/
│   └── v1/
│       ├── best_random_forest.joblib    # Versioned model artifact
│       └── feature_stats.json           # Training distribution stats
├── logs/
│   ├── prediction_logs.csv              # ✅ Append-only prediction log
│   ├── drift_report_<timestamp>.json    # Auto-generated drift reports
│   └── accuracy_report_<timestamp>.json
├── scripts/                    # Helper scripts
├── .env.example                # Environment variable template
├── requirements.txt
├── README.md
├── COMMANDS.md
├── THEORY.md
└── TOPICS-TO-LEARN.md
```

---

## 🚀 Quick Start

### Local (Development)

```bash
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

### Docker (Production)

```bash
# Build image
docker build -t nids-api:v1 .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name nids-api \
  nids-api:v1

# Check it's healthy
curl http://localhost:8000/health
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info + status |
| `GET` | `/health` | Health check + uptime |
| `GET` | `/metrics` | Request counts, latency P95, drift warnings |
| `GET` | `/model/info` | Model version, feature count, class map |
| `POST` | `/predict` | Single prediction with drift check |
| `POST` | `/predict/batch` | Batch predictions |
| `POST` | `/model/reload` | Hot-reload model from disk (zero downtime) |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_12": 1.2, "feature_3": 0.0}}'
```

### Example Response

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

## 📊 Monitoring

### Drift Checker — 4 Modes

```bash
# One-time PSI + KS covariate drift report
python -m monitoring.drift_checker --mode report

# Label distribution / concept drift check
python -m monitoring.drift_checker --mode accuracy

# Continuous watch loop (polls every 60 seconds)
python -m monitoring.drift_checker --mode watch --interval 60

# Compute training baseline stats from raw dataset
python -m monitoring.drift_checker --mode compute-stats \
  --dataset inputs/X_train_final.csv
```

### What Gets Monitored

| Type | Method | Threshold | Alert |
|------|--------|-----------|-------|
| Covariate drift | PSI per feature | PSI > 0.2 = HIGH | `overall_drift: true` |
| Distribution shift | KS test | p < 0.05 | `ks_drift: true` |
| Concept drift | Chi-square on labels | p < 0.05 | `concept_drift: true` |
| Anomaly spike | Benign % drop | < 56% of traffic | `anomaly_flag: true` |
| Per-request drift | Z-score (live) | z > threshold | `drift_warning: true` in response |

---

## ✅ Capstone Deliverables

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| `POST /predict` endpoint | `api.py` → `/predict` route | ✅ |
| Prediction logging | `log_prediction()` → `prediction_logs.csv` | ✅ |
| Request ID tracking | `uuid.uuid4()` per request | ✅ |
| Input validation | Pydantic `PredictRequest` schema | ✅ |
| Versioned model loading | `models/v1/` path via `cfg` | ✅ |
| Data drift detection | PSI + KS in `drift_checker.py` | ✅ |
| Dockerfile | Multi-stage, production-hardened | ✅ |
| `.env.example` | Environment variable template | ✅ |
| Batch prediction | `/predict/batch` endpoint | ✅ |
| Hot model reload | `POST /model/reload` | ✅ |
| Health + metrics | `/health`, `/metrics` endpoints | ✅ |
| P95 latency tracking | `deque(maxlen=1000)` | ✅ |

---

## 🏗️ Architecture

```
Client
  │
  ▼
FastAPI (uvicorn)
  │
  ├── POST /predict ──► prepare_features()
  │                          │
  │                          ├── scaler.transform()
  │                          ├── model.predict()
  │                          ├── model.predict_proba()  → confidence
  │                          ├── check_drift()          → z-score vs feature_stats.json
  │                          └── log_prediction()       → prediction_logs.csv
  │
  ├── GET /metrics  ──► ModelState (in-memory counters + latency deque)
  └── GET /health   ──► ModelState.is_ready + uptime

monitoring/drift_checker.py  (runs independently, reads logs)
  │
  ├── reads prediction_logs.csv
  ├── compares vs feature_stats.json (training baseline)
  ├── PSI per feature + KS test + Chi-square on label distribution
  └── saves logs/drift_report_<timestamp>.json
```

---

## 🔧 Requirements

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0
joblib>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
```

---

## 📌 Key Design Decisions

- **Multi-stage Docker build** — `builder` installs deps, `runtime` copies only what's needed → smaller, safer final image
- **`ModelState` singleton** — all counters, model references, and latency history in one in-memory object; avoids global variable chaos
- **`deque(maxlen=1000)`** — bounded memory for latency ring buffer; never grows unbounded in long-running production service
- **Welford's online algorithm** in `compute_training_stats` — streams huge CSVs in chunks without loading all data into RAM
- **PSI + KS dual detection** — PSI catches gradual distribution shifts, KS catches sudden ones; using both reduces false negatives
- **`asynccontextmanager` lifespan** — replaces deprecated `on_startup`/`on_shutdown` (current FastAPI best practice)
- **Dual-layer drift checking** — z-score per request in `/predict` (real-time) + PSI/KS in `drift_checker.py` (batch analysis)

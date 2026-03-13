# 🚀 Day 5 — ML Model Deployment & MLOps Capstone

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Learning Outcomes](#-learning-outcomes)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Endpoints](#-api-endpoints)
- [API Usage Examples](#-api-usage-examples)
- [Monitoring & Drift Detection](#-monitoring--drift-detection)
- [Docker Deployment](#-docker-deployment)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deliverables Checklist](#-deliverables-checklist)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)

---

## 🎯 Overview

This capstone project demonstrates a **production-ready ML deployment pipeline** with:

- **Real-time model serving** via FastAPI REST API
- **Comprehensive monitoring** for data drift and accuracy decay
- **MLOps best practices** including versioning, logging, and containerization
- **Input validation** using Pydantic schemas
- **Request tracking** with unique IDs for audit trails

### 🔑 Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Server** | FastAPI + Uvicorn | Model serving |
| **Validation** | Pydantic v2 | Input/output schemas |
| **ML Framework** | scikit-learn | Model training |
| **Monitoring** | Custom drift checker | Data drift detection |
| **Containerization** | Docker | Deployment |
| **Logging** | Python logging | Audit trail |

---

## 📚 Learning Outcomes

After completing this project, you will be able to:

✅ Deploy real ML systems in production  
✅ Monitor data drift and accuracy decay  
✅ Build production-ready ML pipelines  
✅ Implement input validation and error handling  
✅ Use Docker for consistent deployments  
✅ Create comprehensive API documentation  

---

## ✨ Features

### Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔌 **REST API** | FastAPI with `/predict` endpoint | ✅ |
| ✅ **Input Validation** | Pydantic schema validation | ✅ |
| 🆔 **Request Tracking** | UUID-based request IDs | ✅ |
| 📝 **Prediction Logging** | CSV audit trail | ✅ |
| 📦 **Model Versioning** | Multiple model versions | ✅ |
| 📊 **Drift Detection** | PSI & KS-test monitoring | ✅ |
| 🐳 **Docker Support** | Multi-stage Dockerfile | ✅ |
| ❤️ **Health Checks** | `/health` endpoint | ✅ |
| 📈 **Metrics** | `/metrics` endpoint | ✅ |
| 🔄 **Hot Reload** | Runtime model reloading | ✅ |
| 📚 **Auto Docs** | Swagger UI & ReDoc | ✅ |

### Monitoring Features

| Feature | Method | Threshold |
|---------|--------|-----------|
| **Covariate Drift** | PSI (Population Stability Index) | > 0.2 |
| **Distribution Shift** | Kolmogorov-Smirnov Test | p < 0.05 |
| **Outlier Detection** | Z-score | > 5.0 |
| **Accuracy Decay** | Label distribution | Configurable |

---

## 📁 Project Structure

```
DAY_5-DEPLOYMENT/
│
├── 📂 deployment/
│   ├── __init__.py
│   ├── api.py                  # 🔌 FastAPI application
│   ├── config.py               # ⚙️ Configuration management
│   ├── logger.py               # 📝 Logging utilities
│   └── schemas.py              # 📋 Pydantic models
│
├── 📂 monitoring/
│   ├── __init__.py
│   └── drift_checker.py        # 📊 Drift detection module
│
├── 📂 models/
│   └── v1/
│       ├── model.joblib        # 🤖 Trained model
│       ├── scaler.joblib       # 📏 Feature scaler
│       └── feature_stats.json  # 📈 Training statistics
│
├── 📂 scripts/
│   ├── train.py                # 🎯 Model training
│   └── generate_mock_logs.py   # 🧪 Test data generator
│
├── 📂 logs/
│   ├── api.log                 # 📝 API logs
│   └── drift_report_*.json     # 📊 Drift reports
│
├── 📂 tests/
│   ├── __init__.py
│   └── test_api.py             # 🧪 API tests
│
├── 📄 .env                     # 🔐 Environment variables
├── 📄 .env.example             # 🔐 Environment template
├── 📄 .gitignore               # 🚫 Git ignore rules
├── 📄 Dockerfile               # 🐳 Container definition
├── 📄 docker-compose.yml       # 🐳 Container orchestration
├── 📄 requirements.txt         # 📦 Dependencies
├── 📄 prediction_logs.csv      # 📝 Prediction audit log
├── 📄 DEPLOYMENT-NOTES.md      # 📋 Deployment docs
└── 📄 README.md                # 📖 This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda
- Docker (optional)
- curl or Postman (for testing)

### Step 1: Clone & Navigate

```bash
cd ~/Documents/HESTABIT_TASKS_3rd_BATCH/WEEK_6-MACHINE_LEARNING_ENGINEERING/DAY_5-DEPLOYMENT
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Environment

```bash
cp .env.example .env
```

### Step 5: Train Model

```bash
python scripts/train.py
```

Expected Output:

```
============================================================
Training Model...
============================================================
Dataset: 5000 samples, 42 features
Validation Accuracy: 0.3520
Saved: models/v1/model.joblib
Saved: models/v1/scaler.joblib
Saved: models/v1/feature_stats.json
============================================================
Training Complete!
============================================================
```

### Step 6: Start API

```bash
python -m deployment.api
```

Expected Output:

```
============================================================
Starting NF-UQ-NIDS API...
============================================================
Loaded model from models/v1/model.joblib
Loaded scaler from models/v1/scaler.joblib
Loaded feature stats (42 features)
Model version: v1
Model loaded: True
API ready to serve requests
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 7: Test API

Open a new terminal and run:

```bash
curl http://localhost:8000/health
```

Expected Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1",
  "uptime_seconds": 10.5,
  "total_requests": 0
}
```

### Step 8: Open Swagger Docs

Open browser: `http://localhost:8000/docs`

---

## 🔌 API Endpoints

### Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root information |
| `/health` | GET | Health check |
| `/metrics` | GET | API metrics |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/reload` | POST | Hot reload model |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc |

### Endpoint Details

#### `GET /health`

Returns service health status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1",
  "uptime_seconds": 120.5,
  "total_requests": 25
}
```

#### `GET /metrics`

Returns API performance metrics.

```json
{
  "total_requests": 100,
  "successful_predictions": 98,
  "failed_predictions": 2,
  "avg_latency_ms": 15.234,
  "p95_latency_ms": 28.5,
  "drift_warnings": 5,
  "uptime_seconds": 3600.0,
  "model_version": "v1"
}
```

#### `POST /predict`

Make a single prediction.

**Request:**

```json
{
  "features": {
    "IN_BYTES": 1500.0,
    "OUT_BYTES": 500.0,
    "PROTOCOL": 6.0,
    "TCP_FLAGS": 24.0,
    "IN_PKTS": 10.0,
    "OUT_PKTS": 5.0
  }
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_version": "v1",
  "prediction": 0,
  "label": "Benign",
  "confidence": 0.92,
  "drift_warning": false,
  "latency_ms": 12.5,
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

#### `POST /predict/batch`

Make batch predictions.

**Request:**

```json
{
  "samples": [
    {"IN_BYTES": 1500.0, "OUT_BYTES": 500.0},
    {"IN_BYTES": 50000.0, "OUT_BYTES": 1000.0},
    {"IN_BYTES": 200.0, "OUT_BYTES": 200.0}
  ]
}
```

---

## 💻 API Usage Examples

### Using cURL

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "IN_BYTES": 1500.0,
      "OUT_BYTES": 500.0,
      "PROTOCOL": 6.0,
      "TCP_FLAGS": 24.0
    }
  }'
```

#### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"IN_BYTES": 100.0, "OUT_BYTES": 50.0},
      {"IN_BYTES": 5000.0, "OUT_BYTES": 1000.0}
    ]
  }'
```

#### Get Metrics

```bash
curl http://localhost:8000/metrics
```

#### Reload Model

```bash
curl -X POST http://localhost:8000/model/reload
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Single prediction
payload = {
    "features": {
        "IN_BYTES": 1500.0,
        "OUT_BYTES": 500.0,
        "PROTOCOL": 6.0
    }
}
response = requests.post(f"{BASE_URL}/predict", json=payload)
print(response.json())

# Batch prediction
payload = {
    "samples": [
        {"IN_BYTES": 100.0, "OUT_BYTES": 50.0},
        {"IN_BYTES": 5000.0, "OUT_BYTES": 1000.0}
    ]
}
response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
print(response.json())
```

### Using HTTPie

```bash
# Health check
http GET localhost:8000/health

# Prediction
http POST localhost:8000/predict features:='{"IN_BYTES": 1500}'
```

---

## 📊 Monitoring & Drift Detection

### Generate Test Logs

```bash
# Generate normal logs
python scripts/generate_mock_logs.py -n 200

# Generate logs with simulated drift
python scripts/generate_mock_logs.py -n 200 --drift
```

### Run Drift Analysis

```bash
# One-time drift report
python -m monitoring.drift_checker --mode report

# Continuous monitoring
python -m monitoring.drift_checker --mode watch --interval 60
```

### Drift Detection Methods

| Method | Description | Alert Threshold |
|--------|-------------|-----------------|
| PSI | Population Stability Index | > 0.2 (High) |
| KS Test | Kolmogorov-Smirnov test | p < 0.05 |
| Z-Score | Standard deviation check | > 5.0 |

### Sample Drift Report

```json
{
  "report_time": "2024-01-15T10:30:00",
  "samples_analyzed": 200,
  "features_drifted": 3,
  "overall_drift": false
}
```

### View Prediction Logs

```bash
# View all logs
cat prediction_logs.csv

# View last 10 predictions
tail -10 prediction_logs.csv

# Count predictions
wc -l prediction_logs.csv
```

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Check status
docker ps
```

### Test Container

```bash
curl http://localhost:8000/health
```

### Stop Container

```bash
docker-compose down
```

### Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker-compose up -d` | Start in background |
| `docker-compose up -d --build` | Rebuild and start |
| `docker-compose logs -f` | View live logs |
| `docker-compose ps` | Check status |
| `docker-compose down` | Stop containers |
| `docker-compose restart` | Restart containers |

### Dockerfile Overview

```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder
# Install dependencies

FROM python:3.11-slim AS runtime
# Copy application
# Expose port 8000
# Run uvicorn
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API host |
| `PORT` | `8000` | API port |
| `WORKERS` | `1` | Uvicorn workers |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_VERSION` | `v1` | Model version |
| `DRIFT_PSI_THRESHOLD` | `0.2` | PSI threshold |
| `DRIFT_ZSCORE_CUTOFF` | `5.0` | Z-score threshold |

### Sample `.env` File

```env
# API Settings
HOST=0.0.0.0
PORT=8000
WORKERS=2
LOG_LEVEL=INFO

# Model Settings
MODEL_VERSION=v1

# Drift Detection
DRIFT_PSI_THRESHOLD=0.2
DRIFT_ZSCORE_CUTOFF=5.0
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=deployment

# Run specific test
pytest tests/test_api.py::test_health -v
```

### Manual Testing

```bash
# Test health endpoint
curl -s http://localhost:8000/health | jq

# Test prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"IN_BYTES": 1500}}' | jq
```

---

## ✅ Deliverables Checklist

### Required Files

| File | Status | Description |
|------|--------|-------------|
| `deployment/api.py` | ✅ | FastAPI application |
| `Dockerfile` | ✅ | Container definition |
| `monitoring/drift_checker.py` | ✅ | Drift detection |
| `prediction_logs.csv` | ✅ | Prediction log |
| `DEPLOYMENT-NOTES.md` | ✅ | Deployment docs |
| `requirements.txt` | ✅ | Dependencies |
| `.env.example` | ✅ | Environment template |
| `README.md` | ✅ | Documentation |

### Required Features

| Feature | Status |
|---------|--------|
| `POST /predict` endpoint | ✅ |
| Prediction logging | ✅ |
| Request ID tracking | ✅ |
| Input validation | ✅ |
| Versioned model loading | ✅ |
| Data drift monitoring | ✅ |
| Accuracy decay detection | ✅ |
| Docker deployment | ✅ |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT                               │
│              (curl / Python / Browser)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP Request
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 MIDDLEWARE LAYER                         ││
│  │              (CORS, Error Handling)                      ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              REQUEST VALIDATION                          ││
│  │              (Pydantic Schemas)                          ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              MODEL INFERENCE                             ││
│  │      ┌─────────┐  ┌─────────┐  ┌──────────┐            ││
│  │      │ Scaler  │→ │  Model  │→ │ Predict  │            ││
│  │      └─────────┘  └─────────┘  └──────────┘            ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              DRIFT DETECTION                             ││
│  │         (Z-score / PSI / KS-test)                        ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              LOGGING & METRICS                           ││
│  │    (prediction_logs.csv / api.log / metrics)             ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   MONITORING LAYER                           │
│  ┌──────────────────┐  ┌────────────────────────────────┐   │
│  │   Drift Checker  │  │      Accuracy Monitor          │   │
│  │   (PSI/KS-test)  │  │  (Label distribution check)    │   │
│  └──────────────────┘  └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Port Already in Use

```bash
# Kill process on port 8000
sudo kill -9 $(sudo lsof -t -i :8000)

# Then restart API
python -m deployment.api
```

### Model Not Found

```bash
# Train model first
python scripts/train.py

# Check model exists
ls -la models/v1/
```

### Module Not Found

```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Docker Permission Error

```bash
# Stop containers
docker-compose down

# Rebuild with no cache
docker-compose build --no-cache

# Start again
docker-compose up -d
```

### Connection Refused

```bash
# Check if API is running
ps aux | grep uvicorn

# Start API
python -m deployment.api
```

---

## 🚀 Future Improvements

- [ ] Add Prometheus metrics exporter
- [ ] Implement Grafana dashboards
- [ ] Add Redis caching layer
- [ ] PostgreSQL for production logging
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] A/B testing support
- [ ] Model explainability (SHAP)
- [ ] Authentication (JWT/OAuth)
- [ ] Rate limiting
- [ ] SSL/TLS support

---

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLOps Principles](https://ml-ops.org/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

---

## 👨‍💻 Author

Day 5 Capstone Project  
Machine Learning Engineering - Week 6  
HESTABIT Tasks 3rd Batch

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  <b>🎉 Congratulations on completing the MLOps Capstone! 🎉</b>
</p>
<p align="center">
  Made with ❤️ for Machine Learning Engineering
</p>

---

## 📋 Quick Command Reference

| Task | Command |
|------|---------|
| Setup | `python -m venv venv && source venv/bin/activate` |
| Install | `pip install -r requirements.txt` |
| Train | `python scripts/train.py` |
| Start API | `python -m deployment.api` |
| Health Check | `curl http://localhost:8000/health` |
| Predict | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": {...}}'` |
| Swagger | Open `http://localhost:8000/docs` |
| Generate Logs | `python scripts/generate_mock_logs.py -n 200` |
| Drift Report | `python -m monitoring.drift_checker --mode report` |
| Docker Start | `docker-compose up -d --build` |
| Docker Logs | `docker-compose logs -f` |
| Docker Stop | `docker-compose down` |
| Stop API | `Ctrl + C` |
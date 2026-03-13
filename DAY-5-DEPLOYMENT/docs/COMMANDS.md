# ⌨️ COMMANDS.md — Day 5 Quick Reference

Every command to run, test, monitor, and debug the Day 5 deployment pipeline.

---

## 🚀 Start the API

```bash
# Development mode (auto-reload on file changes)
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload

# Production mode (multi-worker)
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --workers 4

# Direct Python entry point
python -m deployment.api

# Verify it's up
curl http://localhost:8000/
curl http://localhost:8000/health
```

---

## 📦 Install Dependencies

```bash
# All production dependencies
pip install fastapi uvicorn[standard] pydantic joblib numpy pandas scipy scikit-learn python-dotenv

# Full install from requirements file
pip install -r requirements.txt

# Verify key packages
python -c "import fastapi, uvicorn, pydantic, joblib, scipy; print('All OK')"
```

---

## 🌐 Test API Endpoints

```bash
# Root
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health | python -m json.tool

# Metrics (requests, latency, drift warnings)
curl http://localhost:8000/metrics | python -m json.tool

# Model info
curl http://localhost:8000/model/info | python -m json.tool

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 0.5, "feature_12": 1.2, "feature_3": 0.8}}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"feature_1": 0.5, "feature_12": 1.2},
      {"feature_1": 2.1, "feature_12": 0.3}
    ]
  }'

# Hot reload model (zero downtime)
curl -X POST http://localhost:8000/model/reload | python -m json.tool
```

---

## 🐳 Docker Commands

```bash
# Build production image (multi-stage)
docker build -t nids-api:v1 .

# Build with no cache (clean build)
docker build --no-cache -t nids-api:v1 .

# Run container with volume mounts
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name nids-api \
  nids-api:v1

# Check container logs
docker logs nids-api
docker logs nids-api --follow   # live tail

# Check container health
docker inspect nids-api | python -m json.tool | grep -A5 Health

# Stop and remove
docker stop nids-api && docker rm nids-api

# Shell into running container (debug)
docker exec -it nids-api /bin/bash

# Check image size
docker images nids-api

# Rebuild and restart in one command
docker stop nids-api && docker rm nids-api && \
docker build -t nids-api:v1 . && \
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --name nids-api nids-api:v1
```

---

## 📊 Drift Monitoring Commands

```bash
# Compute baseline feature stats from training data (run once)
python -m monitoring.drift_checker --mode compute-stats \
  --dataset inputs/X_train_final.csv

# Run covariate drift report (PSI + KS test)
python -m monitoring.drift_checker --mode report

# Run concept drift / accuracy check (label distribution)
python -m monitoring.drift_checker --mode accuracy

# Continuous watch loop — alerts every 60 seconds
python -m monitoring.drift_checker --mode watch --interval 60

# Watch with custom interval (e.g., every 5 minutes)
python -m monitoring.drift_checker --mode watch --interval 300

# Run for a specific model version
python -m monitoring.drift_checker --mode report --version v2
```

---

## 📋 Inspect Prediction Logs

```bash
# View raw CSV
cat logs/prediction_logs.csv

# Count total predictions
python -c "
import pandas as pd
df = pd.read_csv('logs/prediction_logs.csv')
print(f'Total predictions: {len(df)}')
print(f'Drift warnings:    {df[\"drift_warning\"].sum()}')
print(f'Avg confidence:    {df[\"confidence\"].mean():.4f}')
print(f'Avg latency (ms):  {df[\"latency_ms\"].mean():.2f}')
"

# Label distribution
python -c "
import pandas as pd
df = pd.read_csv('logs/prediction_logs.csv')
print(df['label'].value_counts())
"

# Latest 10 predictions
python -c "
import pandas as pd
df = pd.read_csv('logs/prediction_logs.csv')
print(df[['request_id','timestamp','label','confidence','drift_warning','latency_ms']].tail(10).to_string())
"

# Find all drift-flagged requests
python -c "
import pandas as pd
df = pd.read_csv('logs/prediction_logs.csv')
drifted = df[df['drift_warning'] == True]
print(f'Drifted requests: {len(drifted)}')
print(drifted[['request_id','timestamp','label','confidence']].to_string())
"

# View latest drift report
ls -t logs/drift_report_*.json | head -1 | xargs cat | python -m json.tool

# View latest accuracy report
ls -t logs/accuracy_report_*.json | head -1 | xargs cat | python -m json.tool
```

---

## 🔥 Load Testing

```bash
# Install load testing tools
pip install httpx asyncio

# Simple load test with Python
python -c "
import httpx, time, json

URL = 'http://localhost:8000/predict'
PAYLOAD = {'features': {'feature_1': 0.5, 'feature_12': 1.2, 'feature_3': 0.8}}
N = 100

start = time.time()
latencies = []
for i in range(N):
    r = httpx.post(URL, json=PAYLOAD)
    latencies.append(r.json()['latency_ms'])

total = time.time() - start
latencies.sort()
print(f'Requests: {N} in {total:.2f}s ({N/total:.1f} req/s)')
print(f'P50: {latencies[int(N*0.5)]:.1f}ms')
print(f'P95: {latencies[int(N*0.95)]:.1f}ms')
print(f'P99: {latencies[int(N*0.99)]:.1f}ms')
"
```

---

## 🛠️ Debug & Troubleshoot

```bash
# Check model file exists
ls -lh models/v1/

# Test model loads correctly outside API
python -c "
import joblib
from pathlib import Path
model = joblib.load('models/v1/best_random_forest.joblib')
print('Model type:', type(model).__name__)
print('Classes:', model.classes_)
print('Features expected:', model.n_features_in_)
"

# Test feature_stats.json is valid
python -c "
import json
with open('models/v1/feature_stats.json') as f:
    stats = json.load(f)
print(f'Features tracked: {len(stats)}')
first = list(stats.items())[0]
print(f'Example: {first[0]} → {first[1]}')
"

# Check environment variables are loaded
python -c "
from deployment.config import cfg
print('Host:', cfg.HOST)
print('Port:', cfg.PORT)
print('Model version:', cfg.MODEL_VERSION)
print('Log path:', cfg.PREDICTION_LOG)
"

# Simulate a drifted request (extreme values)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 999.0, "feature_12": -500.0}}'
# Should return: "drift_warning": true
```

---

## 🗂️ File Locations Summary

| What | Where |
|------|-------|
| Start API | `uvicorn deployment.api:app --reload` |
| Dockerfile | `deployment/Dockerfile` |
| Prediction logs | `logs/prediction_logs.csv` |
| Drift reports | `logs/drift_report_<ts>.json` |
| Accuracy reports | `logs/accuracy_report_<ts>.json` |
| Model artifact | `models/v1/best_random_forest.joblib` |
| Feature stats | `models/v1/feature_stats.json` |
| Env template | `.env.example` |
| Swagger docs | `http://localhost:8000/docs` |

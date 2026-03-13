# 🎯 TOPICS-TO-LEARN.md — Day 5 Deep Dives

Curated learning path based on what you built in the capstone — production APIs, drift monitoring, and MLOps.

---

## ✅ Covered Today (Solidify These)

### 1. FastAPI Production Patterns
**What you built:** Full async API with Pydantic validation, lifespan, CORS, exception handlers  
**Go deeper:**
- [ ] **Dependency injection** — `Depends()` for auth, rate limiting, DB connections
- [ ] **Background tasks** — `BackgroundTasks` to log predictions without blocking response
- [ ] **Middleware** — add request timing, API key auth, or rate limiting as middleware
- [ ] **Response streaming** — for large batch predictions

```python
from fastapi import BackgroundTasks

@app.post("/predict")
async def predict(payload: PredictRequest, background_tasks: BackgroundTasks):
    result = model.predict(...)
    background_tasks.add_task(log_prediction, result)  # doesn't block response
    return result
```

---

### 2. PSI + KS Drift Detection
**What you built:** Per-feature PSI (10-bin histogram ratio) + KS 2-sample test  
**Go deeper:**
- [ ] **CUSUM** (Cumulative Sum) — detects sudden shifts in streaming data
- [ ] **ADWIN** (Adaptive Windowing) — automatically adjusts window size to drift speed
- [ ] **MMD** (Maximum Mean Discrepancy) — kernel-based test for high-dimensional drift
- [ ] **Evidently AI** — open-source library that automates all of the above

```python
pip install evidently

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
report.save_html("drift_report.html")
```

---

### 3. Docker for ML Services
**What you built:** Multi-stage build, health check, volume mounts for models + logs  
**Go deeper:**
- [ ] **Docker Compose** — run API + drift monitor + Redis as one stack
- [ ] **Docker secrets** — proper way to inject API keys (not env vars)
- [ ] **.dockerignore** — exclude `__pycache__`, `.git`, large data files from build context
- [ ] **Non-root user** — run container as non-root for security hardening

```dockerfile
# Add to your Dockerfile (security best practice)
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

---

## 🔶 Next Priority Topics

### 4. MLflow — Experiment Tracking + Model Registry
**Why it matters:** You've trained models across Day 3, 4, 5 with no central record of which is which  
**Topics:**
- [ ] Log Day 3 CV results and Day 4 Optuna trials to MLflow
- [ ] Register `best_random_forest.joblib` in MLflow Model Registry with version tags
- [ ] Transition model through `Staging → Production → Archived` lifecycle
- [ ] Serve directly: `mlflow models serve -m models:/nids-rf/Production`

```python
import mlflow

mlflow.set_experiment("nids-production")
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("test_f1", 0.9850)
    mlflow.sklearn.log_model(model, "model", registered_model_name="nids-rf")
```

---

### 5. Model Monitoring at Scale — Prometheus + Grafana
**Why it matters:** Your `/metrics` endpoint exposes numbers; Prometheus scrapes and Grafana visualises them  
**Topics:**
- [ ] Add `prometheus-fastapi-instrumentator` to auto-expose metrics
- [ ] Track: request rate, error rate, prediction latency histogram, drift warnings/minute
- [ ] Build a Grafana dashboard with alerts (e.g., alert if P95 latency > 100ms)
- [ ] Set up alerting rules — PagerDuty or Slack notification on drift detection

```python
pip install prometheus-fastapi-instrumentator

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
# Now http://localhost:8000/metrics returns Prometheus-formatted data
```

---

### 6. Kubernetes Deployment
**Why it matters:** Docker runs one container; Kubernetes scales to hundreds and handles failures  
**Topics:**
- [ ] Write a `deployment.yaml` and `service.yaml` for your NIDS API
- [ ] Configure horizontal pod autoscaling (HPA) — scale up when CPU > 70%
- [ ] **Liveness vs readiness probes** — liveness restarts crashed pods; readiness gates traffic
- [ ] Rolling updates — deploy v2 alongside v1, shift traffic gradually (same as your version system)

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nids-api
        image: nids-api:v1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
```

---

### 7. CI/CD for ML (GitHub Actions)
**Why it matters:** Manual deployment is error-prone; automate test → build → deploy  
**Topics:**
- [ ] Write a GitHub Actions workflow that runs on every push:
  - `pytest` unit tests for API endpoints
  - `docker build` to verify image builds
  - Push to container registry (Docker Hub / ECR)
  - Deploy to staging, smoke test, promote to production
- [ ] **Model validation gate** — only deploy if new model F1 ≥ current production F1

```yaml
# .github/workflows/deploy.yml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/
  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - run: docker build -t nids-api:${{ github.sha }} .
      - run: docker push your-registry/nids-api:${{ github.sha }}
```

---

### 8. Advanced API Patterns

#### Rate Limiting
```python
pip install slowapi

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, payload: PredictRequest):
    ...
```

#### API Key Authentication
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
```

#### Async Model Inference (for CPU-bound heavy models)
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(payload: PredictRequest):
    # Run CPU-bound inference in thread pool, don't block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, model.predict, vec)
    return result
```

---

## 🔷 Advanced MLOps Topics (Week 2+)

### 9. Feature Stores
- [ ] **Feast** — open-source feature store, separates feature engineering from serving
- [ ] Online store (Redis) for low-latency feature lookup at prediction time
- [ ] Offline store (S3/BigQuery) for training data versioning
- [ ] Point-in-time correct joins — prevents training/serving skew

### 10. Shadow Mode Deployment
- [ ] Route 100% of traffic to model v1 (production)
- [ ] Simultaneously send same requests to v2 (shadow) — no user impact
- [ ] Compare v1 vs v2 predictions on real traffic before promoting v2
- [ ] Your `/predict/batch` endpoint is a building block for this pattern

### 11. Canary Deployments
- [ ] Send 5% of traffic to new model, 95% to old
- [ ] Gradually increase to 10%, 25%, 50%, 100%
- [ ] Auto-rollback trigger: if error rate > 1% or P95 latency > 200ms
- [ ] Implemented with NGINX/Istio traffic splitting or feature flags

### 12. Data Versioning
- [ ] **DVC (Data Version Control)** — git for datasets and models
- [ ] Track which training data produced which model version
- [ ] Reproduce any past experiment exactly: `dvc checkout v1.2`

---

## 📖 Recommended Reading

| Resource | Topic | Priority |
|----------|-------|----------|
| [FastAPI docs](https://fastapi.tiangolo.com/) | Full API reference | ⭐⭐⭐ |
| [Evidently AI docs](https://docs.evidentlyai.com/) | Production drift monitoring | ⭐⭐⭐ |
| [MLflow docs](https://mlflow.org/docs/latest/) | Experiment tracking + registry | ⭐⭐⭐ |
| [Designing ML Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | Production ML architecture | ⭐⭐⭐ |
| [Google MLOps whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) | MLOps maturity model | ⭐⭐ |
| [Prometheus + Grafana tutorial](https://prometheus.io/docs/visualization/grafana/) | Metrics + dashboards | ⭐⭐ |

---

## 🔍 Questions to Answer Before Moving On

1. Your `check_drift()` runs z-score on **every request** — what happens to `drift_warnings` count on the `/metrics` endpoint if a legitimate but unusual traffic spike occurs? Is this a false positive problem?
2. The `deque(maxlen=1000)` stores the last 1000 latencies. If the server handles 10,000 requests/hour, what time window does `maxlen=1000` actually represent? Is P95 still meaningful?
3. The Dockerfile runs as root (`chmod -R 777 logs`). Why is this a security concern, and what would the fix look like?
4. Your `POST /model/reload` has no authentication. What happens if someone calls it repeatedly in production? How would you protect it?
5. The drift watcher uses a `while True` loop with `time.sleep()`. What's the production-grade alternative, and why does it matter for container orchestration?

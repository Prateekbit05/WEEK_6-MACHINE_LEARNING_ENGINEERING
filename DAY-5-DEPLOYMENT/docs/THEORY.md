# 📚 THEORY.md — Day 5: Deployment + Monitoring + MLOps

The theory behind every production decision made in the Day 5 capstone.

---

## 1. FastAPI — Why It's the Standard for ML Serving

FastAPI is built on two foundations: **Starlette** (async web framework) and **Pydantic** (data validation). This makes it ideal for ML APIs:

```
Request JSON
     │
     ▼
Pydantic validates & coerces types   ← automatic, zero boilerplate
     │
     ▼
Your endpoint function (async)
     │
     ▼
Pydantic serialises response         ← type-safe output
     │
     ▼
JSON Response
```

### Why `async` matters for ML APIs
```python
# Synchronous — blocks the entire server during prediction
@app.post("/predict")
def predict(payload: PredictRequest):
    return model.predict(...)  # server frozen here

# Asynchronous — other requests can be handled while I/O waits
@app.post("/predict")
async def predict(payload: PredictRequest):
    return model.predict(...)  # CPU-bound, but non-blocking for I/O
```

Your model inference is CPU-bound (numpy operations), but async still helps because **I/O operations** (reading logs, loading files) don't block other requests.

### Pydantic Schemas — Why Input Validation Matters
Without validation:
```python
# A bad actor sends: {"features": {"feature_1": "hello"}}
vec = np.array([features["feature_1"]])  # crashes inside prediction
# → unhandled 500 error, no logging, no request ID
```

With Pydantic:
```python
class PredictRequest(BaseModel):
    features: Dict[str, float]  # "hello" → 422 Unprocessable Entity automatically
```
Pydantic rejects bad input **before** your code runs — clean error messages, logged cleanly.

---

## 2. The `ModelState` Singleton Pattern

```python
class ModelState:
    def __init__(self):
        self.model = None
        self.total_requests = 0
        self.latencies = deque(maxlen=1000)

state = ModelState()  # one instance for the entire application lifetime
```

**Why a class, not global variables?**
```python
# ❌ Global variables — hard to test, easy to accidentally reset
model = None
total_requests = 0

# ✅ Singleton class — encapsulated, testable, mockable
state = ModelState()
state.total_requests += 1
```

### `deque(maxlen=1000)` for Latency Tracking
```python
from collections import deque
latencies = deque(maxlen=1000)

# When maxlen is reached, oldest item is automatically dropped
# Memory is bounded: always stores at most 1000 floats (~8KB)
# Never need to manually trim the list
```
In a server running for days, an unbounded list would grow until OOM. `deque(maxlen=N)` is the correct data structure for a sliding window of metrics.

---

## 3. Versioned Model Loading

```
models/
└── v1/
    ├── best_random_forest.joblib
    ├── feature_stats.json
    └── scaler.joblib   (optional)
```

**Why version the model directory?**
```python
# cfg.MODEL_VERSION = "v1" (from .env)
model_path = cfg.MODEL_DIR / cfg.MODEL_VERSION / "best_random_forest.joblib"

# To deploy v2: change .env → MODEL_VERSION=v2, call POST /model/reload
# Old version still on disk for rollback
```

This pattern enables:
- **Zero-downtime deployment** — `POST /model/reload` swaps the model in memory
- **Instant rollback** — change env var back to `v1`, reload
- **A/B testing** — run two containers with different `MODEL_VERSION` values

### `joblib` vs `pickle` for Production
```python
# pickle — general Python serialization
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)      # stores every numpy array as Python bytes

# joblib — memory-mapped numpy arrays
import joblib
joblib.dump(model, 'model.joblib')  # numpy arrays stored as raw binary
joblib.load('model.joblib')         # uses mmap: no full RAM copy on load
```
For a Random Forest with 292 trees, joblib loads 5–10× faster and uses less RAM.

---

## 4. Drift Detection — Three Types, Three Methods

### Type 1: Covariate Drift (Feature Distribution Shift)
The input features `X` change distribution. The model was trained on 2025 traffic; 2026 traffic looks different.

**Detection: Population Stability Index (PSI)**
```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

PSI < 0.1   → No significant drift (LOW)
PSI 0.1–0.2 → Some shift, monitor (MEDIUM)
PSI > 0.2   → Significant drift, investigate (HIGH)
```

Your code computes PSI per feature using 10 equal-width bins:
```python
bins = np.linspace(min_val, max_val, 11)  # 10 buckets
e_pct = histogram(training_data, bins) / total
a_pct = histogram(production_data, bins) / total
psi = sum((a_pct - e_pct) * log(a_pct / e_pct))
```

**Detection: Kolmogorov-Smirnov Test**
```
KS statistic = max|F_train(x) - F_prod(x)|
             = maximum vertical distance between CDFs

p < 0.05 → distributions are significantly different
```
PSI catches gradual drift; KS catches sudden distributional changes. Your system uses **both** to reduce false negatives.

### Type 2: Concept Drift (The relationship between X and y changes)
The same features now mean something different. A port scan that was "Benign" in 2024 is now a known attack pattern.

**Detection: Chi-square test on label distribution**
```python
# Compare recent label counts vs historical
chi_stat, p = chisquare(f_obs=recent_counts, f_exp=expected_counts)
# p < 0.05 → label distribution has significantly shifted
```

### Type 3: Real-time Z-score check (per request)
```python
z_score = abs((incoming_value - training_mean) / training_std)
if z_score > threshold:
    drift_warning = True  # returned in API response
```
This runs on **every single prediction** — instant feedback, no batch needed.

```
                     Drift Detection Strategy
                     ─────────────────────────

Per-request:  Z-score ──────────────────► drift_warning in response (real-time)

Batch:        PSI + KS ─────────────────► drift_report_<ts>.json (scheduled)
              Chi-square on labels ──────► accuracy_report_<ts>.json
```

---

## 5. Welford's Online Algorithm

Computing mean and variance requires seeing all data. For your 200k+ row CSV, loading everything at once is wasteful.

**Welford's algorithm — one-pass, numerically stable:**
```python
for each new value x:
    n += 1
    delta = x - mean
    mean += delta / n           # running mean
    M2 += delta * (x - mean)   # running sum of squares

variance = M2 / (n - 1)
std = sqrt(variance)
```

**Why not the naive formula?**
```
Naive: var = (Σx²)/n - mean²   ← catastrophic cancellation for large x values
Welford: numerically stable for any magnitude of x
```

Your `drift_checker.py` combines this with `chunksize=200_000` pandas reading — processes unlimited file sizes with constant RAM.

---

## 6. Multi-Stage Docker Build

```dockerfile
# Stage 1: Builder — has gcc, g++, build tools
FROM python:3.11-slim AS builder
RUN pip install --prefix=/install -r requirements.txt
# All compiled .so files and packages now in /install

# Stage 2: Runtime — lean final image
FROM python:3.11-slim AS runtime
COPY --from=builder /install /usr/local  # only the installed packages
COPY deployment/ ./deployment/            # only application code
# build tools (gcc, g++) NOT in final image
```

**Result:**
```
Without multi-stage:  ~1.2 GB (includes gcc, build headers, cache)
With multi-stage:     ~400 MB (python slim + packages only)
```

Security benefit: the attack surface is smaller — no compiler tools, no build cache in the running container.

### Health Check in Dockerfile
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```
Docker (and Kubernetes) polls this every 30 seconds. If it fails 3 times: container is marked `unhealthy` and orchestrators restart it automatically.

---

## 7. The Prediction Log — Append-Only CSV

```
request_id | timestamp | model_version | prediction | label | confidence | drift_warning | latency_ms | features_json
```

**Why CSV (not a database)?**
- Simple, portable, readable with any tool
- Append-only writes are fast (no indexing overhead)
- Can be ingested into any monitoring system (Grafana, Prometheus, S3)
- `features_json` stores the full input for post-hoc analysis and drift checking

**Why `uuid4()` for request IDs?**
```python
request_id = str(uuid.uuid4())  # "a3f1c2d4-7b8e-4f91-a2c3-d4e5f6a7b8c9"
```
- Globally unique — no coordination required across multiple API workers
- Enables end-to-end request tracing from client → log → drift report
- Deterministically link a drift warning to the exact input that caused it

---

## 8. The `asynccontextmanager` Lifespan Pattern

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    load_model()
    ensure_log_csv()
    
    yield  # ← application runs here
    
    # Shutdown code
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

This replaced the old `@app.on_event("startup")` decorator (deprecated in FastAPI 0.93+). The `yield` cleanly separates startup from shutdown, and the context manager pattern means the shutdown code always runs — even on crashes.

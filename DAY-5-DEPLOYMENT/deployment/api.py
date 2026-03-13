"""
=============================================================================
Day 5 — Production FastAPI for ML Model Serving
=============================================================================

Features:
- POST /predict endpoint with input validation
- Prediction logging with request ID tracking
- Versioned model loading
- Data drift detection
- Health checks and metrics
- CORS support
=============================================================================
"""

import csv
import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment.config import cfg
from deployment.logger import setup_logging, get_logger
from deployment.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

# Setup logging
setup_logging(cfg.LOG_DIR, cfg.LOG_LEVEL)
logger = get_logger("nids-api")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATE & METRICS
# ─────────────────────────────────────────────────────────────────────────────

class ModelState:
    """Singleton for managing model state and metrics."""
    
    def __init__(self):
        self.model: Any = None
        self.scaler: Any = None
        self.feature_stats: Dict = {}
        self.version: str = cfg.MODEL_VERSION
        self.loaded_at: Optional[str] = None
        
        # Metrics
        self.start_time: float = time.time()
        self.total_requests: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.drift_warnings: int = 0
        self.latencies: deque = deque(maxlen=1000)  # Keep last 1000 latencies
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


state = ModelState()

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> bool:
    """Load model and associated artifacts."""
    model_path = cfg.get_model_path()
    scaler_path = cfg.get_scaler_path()
    stats_path = cfg.get_stats_path()
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        state.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        if scaler_path.exists():
            state.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        if stats_path.exists():
            with open(stats_path) as f:
                state.feature_stats = json.load(f)
            logger.info(f"Loaded feature stats ({len(state.feature_stats)} features)")
        
        state.loaded_at = datetime.utcnow().isoformat()
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def ensure_log_csv():
    """Create prediction log CSV with headers if not exists."""
    if not cfg.PREDICTION_LOG.exists():
        cfg.PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.PREDICTION_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "timestamp", "model_version",
                "prediction", "label", "confidence",
                "drift_warning", "latency_ms", "features_json"
            ])
        logger.info(f"Created prediction log: {cfg.PREDICTION_LOG}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("=" * 60)
    logger.info("Starting NF-UQ-NIDS API...")
    logger.info("=" * 60)
    
    load_model()
    ensure_log_csv()
    
    logger.info(f"Model version: {state.version}")
    logger.info(f"Model loaded: {state.is_ready}")
    logger.info("API ready to serve requests")
    
    yield
    
    logger.info("Shutting down API...")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NF-UQ-NIDS-v2 API",
    description="Network Intrusion Detection System API with MLOps features",
    version=cfg.MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_drift(features: Dict[str, float]) -> bool:
    """
    Check for data drift using z-score method.
    Returns True if drift is detected.
    """
    if not state.feature_stats:
        return False
    
    drift_count = 0
    total_checked = 0
    
    for col, val in features.items():
        if col in state.feature_stats:
            stats = state.feature_stats[col]
            mean = stats.get("mean", 0)
            std = stats.get("std", 1)
            
            if std > 0:
                z_score = abs((val - mean) / std)
                if z_score > cfg.DRIFT_ZSCORE_CUTOFF:
                    drift_count += 1
            total_checked += 1
    
    if total_checked == 0:
        return False
    
    drift_ratio = drift_count / total_checked
    return drift_ratio > cfg.DRIFT_OUTLIER_RATIO


def log_prediction(
    request_id: str,
    prediction: int,
    label: str,
    confidence: float,
    drift_warning: bool,
    latency_ms: float,
    features: Dict[str, float],
) -> None:
    """Log prediction to CSV file."""
    try:
        with open(cfg.PREDICTION_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                request_id,
                datetime.utcnow().isoformat(),
                state.version,
                prediction,
                label,
                round(confidence, 6),
                drift_warning,
                round(latency_ms, 3),
                json.dumps(features),
            ])
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


def prepare_features(features: Dict[str, float]) -> np.ndarray:
    """Prepare feature vector for prediction."""
    vec = np.array(
        [features.get(c, 0.0) for c in cfg.FEATURE_COLS],
        dtype=np.float32
    ).reshape(1, -1)
    
    if state.scaler is not None:
        vec = state.scaler.transform(vec)
    
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": "NF-UQ-NIDS-v2 API",
        "version": state.version,
        "status": "healthy" if state.is_ready else "degraded",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.is_ready else "degraded",
        model_loaded=state.is_ready,
        model_version=state.version,
        uptime_seconds=round(state.uptime, 2),
        total_requests=state.total_requests,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Health"])
async def metrics():
    """Get API metrics."""
    return MetricsResponse(
        total_requests=state.total_requests,
        successful_predictions=state.successful_predictions,
        failed_predictions=state.failed_predictions,
        avg_latency_ms=round(state.avg_latency, 3),
        p95_latency_ms=round(state.p95_latency, 3),
        drift_warnings=state.drift_warnings,
        uptime_seconds=round(state.uptime, 2),
        model_version=state.version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    model_type = type(state.model).__name__ if state.model else "None"
    
    return ModelInfoResponse(
        version=state.version,
        loaded_at=state.loaded_at,
        features_count=len(cfg.FEATURE_COLS),
        classes=cfg.LABEL_MAP,
        model_type=model_type,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(payload: PredictRequest):
    """
    Make a prediction for network traffic classification.
    
    - **features**: Dictionary of feature names to values
    
    Returns prediction with confidence score and drift warning.
    """
    state.total_requests += 1
    
    if not state.is_ready:
        state.failed_predictions += 1
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    
    try:
        # Prepare features
        vec = prepare_features(payload.features)
        
        # Make prediction
        prediction = int(state.model.predict(vec)[0])
        label = cfg.LABEL_MAP.get(prediction, f"Class-{prediction}")
        
        # Get confidence
        if hasattr(state.model, "predict_proba"):
            proba = state.model.predict_proba(vec)[0]
            confidence = float(proba.max())
        else:
            confidence = 1.0
        
        # Check for drift
        drift_warning = check_drift(payload.features)
        if drift_warning:
            state.drift_warnings += 1
        
        latency_ms = (time.perf_counter() - t0) * 1000
        state.latencies.append(latency_ms)
        state.successful_predictions += 1
        
        # Log prediction
        log_prediction(
            request_id=request_id,
            prediction=prediction,
            label=label,
            confidence=confidence,
            drift_warning=drift_warning,
            latency_ms=latency_ms,
            features=payload.features,
        )
        
        return PredictResponse(
            request_id=request_id,
            model_version=state.version,
            prediction=prediction,
            label=label,
            confidence=round(confidence, 6),
            drift_warning=drift_warning,
            latency_ms=round(latency_ms, 3),
            timestamp=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        state.failed_predictions += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(payload: BatchPredictRequest):
    """Make batch predictions."""
    state.total_requests += 1
    
    if not state.is_ready:
        state.failed_predictions += 1
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    
    predictions = []
    for sample in payload.samples:
        vec = prepare_features(sample)
        pred = int(state.model.predict(vec)[0])
        label = cfg.LABEL_MAP.get(pred, f"Class-{pred}")
        
        if hasattr(state.model, "predict_proba"):
            confidence = float(state.model.predict_proba(vec)[0].max())
        else:
            confidence = 1.0
        
        predictions.append({
            "prediction": pred,
            "label": label,
            "confidence": round(confidence, 6),
        })
    
    latency_ms = (time.perf_counter() - t0) * 1000
    state.latencies.append(latency_ms)
    state.successful_predictions += 1
    
    return BatchPredictResponse(
        request_id=request_id,
        model_version=state.version,
        predictions=predictions,
        total_samples=len(predictions),
        avg_latency_ms=round(latency_ms / len(predictions), 3),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk."""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded", "version": state.version}
    raise HTTPException(status_code=500, detail="Failed to reload model")


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "deployment.api:app",
        host=cfg.HOST,
        port=cfg.PORT,
        reload=cfg.RELOAD,
        workers=cfg.WORKERS,
        log_level=cfg.LOG_LEVEL.lower(),
    )
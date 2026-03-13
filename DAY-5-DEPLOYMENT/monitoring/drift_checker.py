"""
=============================================================================
Day 5 — Drift & Accuracy Monitor
=============================================================================

Usage:
  python -m monitoring.drift_checker --mode report
  python -m monitoring.drift_checker --mode accuracy
  python -m monitoring.drift_checker --mode watch --interval 60
  python -m monitoring.drift_checker --mode compute-stats --dataset path/to/data.csv
=============================================================================
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deployment.config import cfg
from deployment.logger import setup_logging, get_logger

# Setup logging
setup_logging(cfg.LOG_DIR, cfg.LOG_LEVEL)
logger = get_logger("drift-checker")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PSI_THRESHOLDS = {"low": 0.1, "medium": 0.2}  # >0.2 → high drift
KS_ALPHA = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Compute Training Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_training_stats(
    dataset_path: Path,
    version: str = "v1",
    chunk_size: int = 200_000,
) -> Dict:
    """
    Compute per-feature statistics from training data.
    Uses chunked reading for large files.
    """
    logger.info(f"Computing training stats from {dataset_path}")
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return {}
    
    # Online statistics computation (Welford's algorithm)
    n = {c: 0 for c in cfg.FEATURE_COLS}
    mean = {c: 0.0 for c in cfg.FEATURE_COLS}
    M2 = {c: 0.0 for c in cfg.FEATURE_COLS}
    col_min = {c: np.inf for c in cfg.FEATURE_COLS}
    col_max = {c: -np.inf for c in cfg.FEATURE_COLS}
    
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        
        for col in cfg.FEATURE_COLS:
            if col not in chunk.columns:
                continue
            
            vals = pd.to_numeric(chunk[col], errors="coerce").dropna().values
            
            for x in vals:
                n[col] += 1
                delta = x - mean[col]
                mean[col] += delta / n[col]
                M2[col] += delta * (x - mean[col])
            
            if len(vals) > 0:
                col_min[col] = min(col_min[col], vals.min())
                col_max[col] = max(col_max[col], vals.max())
    
    # Build stats dictionary
    stats_dict: Dict[str, Dict] = {}
    for col in cfg.FEATURE_COLS:
        if n[col] > 0:
            std = np.sqrt(M2[col] / max(n[col] - 1, 1))
            stats_dict[col] = {
                "mean": round(float(mean[col]), 6),
                "std": round(float(std), 6),
                "min": round(float(col_min[col]) if col_min[col] != np.inf else 0.0, 6),
                "max": round(float(col_max[col]) if col_max[col] != -np.inf else 0.0, 6),
                "n": n[col],
            }
    
    # Save stats
    out_path = cfg.MODEL_DIR / version / "feature_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    
    logger.info(f"Saved feature stats ({len(stats_dict)} features) → {out_path}")
    return stats_dict


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PSI & KS Drift Detection
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    eps = 1e-8
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    
    if mx == mn:
        return 0.0
    
    bins = np.linspace(mn, mx, buckets + 1)
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=bins)
    
    e_pct = e_counts / (e_counts.sum() + eps)
    a_pct = a_counts / (a_counts.sum() + eps)
    
    e_pct = np.where(e_pct == 0, eps, e_pct)
    a_pct = np.where(a_pct == 0, eps, a_pct)
    
    psi_val = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi_val)


def ks_test(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float, bool]:
    """Perform Kolmogorov-Smirnov test."""
    stat, p = stats.ks_2samp(expected, actual)
    return float(stat), float(p), bool(p < KS_ALPHA)


def run_drift_report(version: str = "v1", min_samples: int = 100) -> Dict:
    """
    Generate drift report comparing prediction logs to training stats.
    """
    stats_path = cfg.get_stats_path()
    
    if not stats_path.exists():
        logger.error("No feature_stats.json found. Run --mode compute-stats first.")
        return {"error": "missing_stats"}
    
    if not cfg.PREDICTION_LOG.exists():
        logger.warning("No prediction log found")
        return {"error": "no_logs"}
    
    # Load data
    df = pd.read_csv(cfg.PREDICTION_LOG)
    
    if len(df) < min_samples:
        logger.info(f"Only {len(df)} samples. Need {min_samples} for reliable drift check.")
        return {"samples": len(df), "sufficient": False}
    
    with open(stats_path) as f:
        train_stats = json.load(f)
    
    # Parse features from logs
    try:
        if "features_json" in df.columns:
            feat_df = pd.json_normalize(df["features_json"].apply(json.loads))
        else:
            logger.warning("No features_json column in logs")
            return {"error": "missing_features_column"}
    except Exception as e:
        logger.error(f"Cannot parse features_json: {e}")
        return {"error": str(e)}
    
    # Analyze each feature
    results = {}
    n_drifted = 0
    
    for col in cfg.FEATURE_COLS:
        if col not in feat_df.columns or col not in train_stats:
            continue
        
        prod_vals = feat_df[col].dropna().values.astype(float)
        if len(prod_vals) < 10:
            continue
        
        t_mean = train_stats[col]["mean"]
        t_std = max(train_stats[col]["std"], 1e-8)
        
        # Simulate training distribution
        rng = np.random.default_rng(42)
        train_sim = rng.normal(t_mean, t_std, size=5000)
        
        # Compute metrics
        psi_val = compute_psi(train_sim, prod_vals)
        ks_stat, ks_p, ks_drift = ks_test(train_sim, prod_vals)
        
        if psi_val >= PSI_THRESHOLDS["medium"] or ks_drift:
            n_drifted += 1
        
        severity = (
            "HIGH" if psi_val >= PSI_THRESHOLDS["medium"] else
            "MEDIUM" if psi_val >= PSI_THRESHOLDS["low"] else
            "LOW"
        )
        
        results[col] = {
            "psi": round(psi_val, 6),
            "psi_severity": severity,
            "ks_stat": round(ks_stat, 6),
            "ks_p": round(ks_p, 6),
            "ks_drift": ks_drift,
            "prod_mean": round(float(prod_vals.mean()), 4),
            "train_mean": round(t_mean, 4),
        }
    
    drift_fraction = n_drifted / max(len(results), 1)
    
    report = {
        "report_time": datetime.utcnow().isoformat(),
        "model_version": version,
        "samples_analyzed": len(df),
        "features_checked": len(results),
        "features_drifted": n_drifted,
        "drift_fraction": round(drift_fraction, 4),
        "overall_drift": drift_fraction > 0.20,
        "features": results,
    }
    
    # Save report
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.LOG_DIR / f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Drift report saved → {out}")
    logger.info(f"Drifted features: {n_drifted}/{len(results)} | Overall drift: {report['overall_drift']}")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Accuracy / Label Distribution Check
# ═══════════════════════════════════════════════════════════════════════════════

def run_accuracy_check(version: str = "v1", window_hours: int = 24) -> Dict:
    """
    Check for concept drift through label distribution changes.
    """
    if not cfg.PREDICTION_LOG.exists():
        return {"error": "no_logs"}
    
    df = pd.read_csv(cfg.PREDICTION_LOG, parse_dates=["timestamp"])
    
    if df.empty:
        return {"error": "empty_logs"}
    
    # Split into recent and older
    cutoff = df["timestamp"].max() - timedelta(hours=window_hours)
    recent = df[df["timestamp"] >= cutoff]
    older = df[df["timestamp"] < cutoff]
    
    if len(recent) < 10:
        return {"warning": "insufficient_recent_data", "samples": len(recent)}
    
    # Label distributions
    recent_dist = (recent["label"].value_counts() / len(recent)).to_dict()
    older_dist = (older["label"].value_counts() / len(older)).to_dict() if len(older) > 0 else {}
    
    # Check for anomalies
    benign_pct = recent_dist.get("Benign", 0.0)
    expected_benign = 0.80
    anomaly_flag = benign_pct < expected_benign * 0.7  # 30% tolerance
    
    # Chi-square test
    chi_result = None
    if older_dist:
        labels = list(set(recent_dist) | set(older_dist))
        obs_r = np.array([recent_dist.get(l, 0) * len(recent) for l in labels])
        obs_o = np.array([older_dist.get(l, 0) * len(older) for l in labels])
        
        if obs_r.sum() > 0 and obs_o.sum() > 0:
            chi_stat, chi_p = stats.chisquare(
                f_obs=obs_r + 1,  # Laplace smoothing
                f_exp=(obs_o / obs_o.sum()) * obs_r.sum() + 1,
            )
            chi_result = {
                "chi2": round(float(chi_stat), 4),
                "p_value": round(float(chi_p), 6),
                "concept_drift": bool(chi_p < 0.05),
            }
    
    result = {
        "report_time": datetime.utcnow().isoformat(),
        "model_version": version,
        "window_hours": window_hours,
        "recent_samples": len(recent),
        "older_samples": len(older),
        "recent_distribution": {k: round(v, 4) for k, v in recent_dist.items()},
        "benign_fraction": round(benign_pct, 4),
        "anomaly_flag": anomaly_flag,
        "chi_square_test": chi_result,
        "avg_confidence": round(float(recent["confidence"].mean()), 4),
        "avg_latency_ms": round(float(recent["latency_ms"].mean()), 2),
    }
    
    # Save report
    out = cfg.LOG_DIR / f"accuracy_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Accuracy report saved → {out}")
    logger.info(f"Benign: {benign_pct:.1%} | Anomaly: {anomaly_flag}")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Continuous Watch Loop
# ═══════════════════════════════════════════════════════════════════════════════

def watch_loop(version: str = "v1", interval: int = 60) -> None:
    """Continuously monitor for drift."""
    logger.info(f"Starting drift watcher — polling every {interval}s")
    
    while True:
        logger.info("─" * 50)
        logger.info("Running drift & accuracy check...")
        
        try:
            drift_r = run_drift_report(version=version, min_samples=50)
            acc_r = run_accuracy_check(version=version)
            
            if drift_r.get("overall_drift"):
                logger.critical("🚨 COVARIATE DRIFT DETECTED 🚨")
            
            if acc_r.get("anomaly_flag"):
                logger.critical("🚨 LABEL DISTRIBUTION ANOMALY 🚨")
            
            if chi := acc_r.get("chi_square_test"):
                if chi.get("concept_drift"):
                    logger.critical("🚨 CONCEPT DRIFT DETECTED 🚨")
                    
        except Exception as e:
            logger.error(f"Watch error: {e}")
        
        time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Drift & Accuracy Monitor")
    parser.add_argument(
        "--mode",
        choices=["report", "accuracy", "watch", "compute-stats"],
        default="report",
        help="Operation mode",
    )
    parser.add_argument("--version", default="v1", help="Model version")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval (seconds)")
    parser.add_argument("--dataset", default=None, help="Dataset path for compute-stats")
    parser.add_argument("--chunk", type=int, default=200_000, help="Chunk size for CSV")
    
    args = parser.parse_args()
    
    if args.mode == "report":
        result = run_drift_report(version=args.version)
        print(json.dumps(result, indent=2))
    
    elif args.mode == "accuracy":
        result = run_accuracy_check(version=args.version)
        print(json.dumps(result, indent=2))
    
    elif args.mode == "watch":
        watch_loop(version=args.version, interval=args.interval)
    
    elif args.mode == "compute-stats":
        if not args.dataset:
            parser.error("--dataset required for compute-stats mode")
        compute_training_stats(
            dataset_path=Path(args.dataset),
            version=args.version,
            chunk_size=args.chunk,
        )


if __name__ == "__main__":
    main()
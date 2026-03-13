# 📦 Week 6 — Day 2: Feature Engineering + Feature Selection
### Project: Network Intrusion Detection System (NF-UQ-NIDS-v2)

---

## 🎯 What This Day Covers

Day 2 transforms the cleaned 42-column dataset from Day 1 into a rich **144-feature engineered dataset**, then intelligently reduces it to the **50 most informative features** using a 6-method ensemble selection pipeline. The result — `X_train_final.csv` and `X_test_final.csv` — is ready for model training on Day 3.

---

## 📁 Project Structure

```
src/
├── features/
│   ├── build_features.py              # FeatureEngineer — creates 100+ new features
│   ├── feature_selector.py            # FeatureSelector — selects best 50 features
│   ├── transformers.py                # Transformer classes (Log, Sqrt, Power, etc.)
│   ├── feature_list.json              # Final 50 features + all importance scores
│   └── engineered_features.json       # Engineering step metadata
├── data/
│   └── processed/
│       ├── X_train_engineered.csv     # Post-engineering (144 features)
│       ├── X_test_engineered.csv
│       └── final_features/
│           ├── X_train_final.csv      # Post-selection (50 features)
│           └── X_test_final.csv
└── notebooks/
    └── plots/
        ├── importance_mutual_information.png
        ├── importance_tree_based.png
        ├── importance_gradient_boosting.png
        ├── feature_correlation.png
        └── feature_selection_votes.png
```

---

## 📊 Pipeline Results

| Stage | Features | Action |
|---|---|---|
| Input from Day 1 | 42 | Cleaned numerical features |
| After Engineering | 144 | +102 new features created |
| After Variance Filter | — | Removed near-zero variance |
| After Correlation Filter | — | Removed corr > 0.95 pairs |
| **Final Output** | **50** | Ensemble voting selection |

### New Features by Transformation Type

| Type | Count | Example |
|---|---|---|
| Log transforms | 10 | `L4_SRC_PORT_log`, `IN_BYTES_log` |
| Sqrt transforms | 10 | `IN_BYTES_sqrt`, `OUT_BYTES_sqrt` |
| Power transforms | 5 | `PROTOCOL_pow2` |
| Interaction features | 15 | `L4_SRC_PORT_x_PROTOCOL` |
| Ratio features | 15 | `L4_SRC_PORT_div_IN_BYTES` |
| Aggregation features | 6 | `agg_sum`, `agg_skew`, `agg_median` |
| Binning features | 10 | `OUT_BYTES_binned`, `TCP_FLAGS_binned` |
| Statistical features | 10+ | `IN_BYTES_percentile` |

### Top 5 Features by Mutual Information

| Feature | MI Score | Why It Matters |
|---|---|---|
| `OUT_BYTES_binned` | 0.0372 | Binned byte count captures traffic volume categories |
| `L4_SRC_PORT_x_PROTOCOL` | 0.0365 | Port × protocol interaction flags unusual combinations |
| `OUT_PKTS_binned` | 0.0366 | Packet count bins distinguish DDoS vs normal flows |
| `L4_SRC_PORT_log` | 0.0347 | Log-scaled port compresses high ephemeral port range |
| `agg_sum` | 0.0335 | Row-level feature sum captures overall flow intensity |

---

## 🚀 How to Run

### Step 1 — Feature Engineering
```bash
python src/features/build_features.py
```

### Step 2 — Feature Selection
```bash
python src/features/feature_selector.py
```

### Step 3 — Run both in sequence
```bash
python src/features/build_features.py && python src/features/feature_selector.py
```

---

## ⚙️ Configuration (config.yaml)

```yaml
feature_engineering:
  transformations:
    log:   { n_features: 10 }
    sqrt:  { n_features: 10 }
    power: { n_features: 5, power: 2 }
  interactions:  { max_interactions: 15, top_features: 6 }
  binning:       { enabled: true, n_bins: 5, strategy: quantile }
  aggregations:  { enabled: true }
  statistical:   { enabled: true }

feature_selection:
  n_features: 50
  variance:    { threshold: 0.01 }
  correlation: { threshold: 0.95 }
  voting:      { min_votes: 2 }
  rfe:         { enabled: true, n_features_to_select: 30 }

scaling:
  method: standard
```

---

## ✅ Deliverables Checklist

- [x] `features/build_features.py` — FeatureEngineer class, 8 transformer types
- [x] `features/feature_selector.py` — FeatureSelector class, 6 selection methods
- [x] `features/feature_list.json` — 50 features with full importance scores
- [x] `FEATURE-ENGINEERING-DOC.md` — Summary documentation

---

## 🔑 Key Design Decisions

- **Transformers fit on train, applied to test** — zero data leakage across all 8 transform types
- **Ensemble voting** (≥2 method agreement) — more robust than trusting a single selector
- **RFE pre-filtered to top 100** by variance — keeps RFE computation tractable on 144 features
- **Scaling deferred to Day 3** — each model type (tree vs linear) applies its preferred scaler
- **Target encoded before selection, decoded before saving** — final CSV contains original class labels

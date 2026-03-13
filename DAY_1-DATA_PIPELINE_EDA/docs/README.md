# 📦 Week 6 — Day 1: Data Pipeline + EDA
### Project: Network Intrusion Detection System (NF-UQ-NIDS-v2)

---

## 🎯 What This Day Covers

This day builds the **complete data foundation** for an ML-based network intrusion detection system. The pipeline ingests raw network flow data, cleans it, handles class imbalance, splits it for training, and produces a full EDA report — all production-ready and config-driven.

---

## 📁 Project Structure

```
src/
├── data/
│   ├── raw/                  # Original NF-UQ-NIDS-v2 dataset
│   ├── processed/
│   │   ├── final.csv         # Cleaned, combined dataset
│   │   ├── train.csv         # Training split
│   │   ├── test.csv          # Test split
│   │   ├── sample.csv        # 100k-row working sample
│   │   ├── feature_info.json # Feature metadata
│   │   └── pipeline_results.json
│   └── metadata/             # Data version hashes
├── pipelines/
│   └── data_pipeline.py      # Main pipeline (DataPipeline class)
├── notebooks/
│   └── EDA.ipynb             # Full exploratory analysis
├── utils/
│   ├── logger.py
│   └── helpers.py
├── config/
│   └── config.yaml
└── logs/
```

---

## 📊 Dataset Summary

| Property | Value |
|---|---|
| Dataset | NF-UQ-NIDS-v2 (Network Flow) |
| Working Sample | 100,000 rows |
| Total Features | 46 columns (42 numerical, 4 categorical) |
| Target Column | `Attack` |
| Number of Classes | 20 attack types |
| Missing Values | 0 |
| Duplicates | 0 |
| Class Imbalance Ratio | 32,986 : 1 |

### Attack Class Distribution (Top 5)
| Class | Count | % |
|---|---|---|
| Benign | 32,986 | 33.0% |
| DDoS | 28,742 | 28.7% |
| DoS | 23,542 | 23.5% |
| Scanning | 4,955 | 5.0% |
| Reconnaissance | 3,444 | 3.4% |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place raw dataset
```bash
# Put your CSV in:
src/data/raw/data.csv
```

### 3. Run the full pipeline
```bash
python src/pipelines/data_pipeline.py
```

### 4. Run EDA notebook
```bash
jupyter notebook src/notebooks/EDA.ipynb
```

---

## ⚙️ Configuration

All pipeline behaviour is controlled through `src/config/config.yaml`:

```yaml
data:
  raw_path: src/data/raw/data.csv
  sample_size: 100000
  test_size: 0.2
  stratify: true

preprocessing:
  missing_values:
    numerical_strategy: median
  outliers:
    detection_method: iqr
    treatment: cap
  imbalance:
    handle: true
    method: smote
```

---

## ✅ Deliverables Checklist

- [x] `data_pipeline.py` — Production pipeline with chunked loading
- [x] `EDA.ipynb` — Full exploratory analysis notebook
- [x] `data/processed/final.csv` — Cleaned dataset
- [x] `DATA-REPORT.md` — EDA findings and recommendations

---

## 🔑 Key Design Decisions

- **Chunked loading** (50k rows/chunk) handles files larger than RAM
- **SMOTE** applied only on training data to prevent data leakage
- **IQR-based outlier capping** preserves data volume vs. removal
- **Dataframe hashing** for lightweight dataset versioning
- **Stratified split** maintains class proportions in train/test sets
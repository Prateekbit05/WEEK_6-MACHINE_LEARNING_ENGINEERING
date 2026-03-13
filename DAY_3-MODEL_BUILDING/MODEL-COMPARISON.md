# Model Comparison Report

Generated on: 2026-02-16 18:46:59

## Dataset
NF-UQ-NIDS-v2 (200k rows sample)

## Cross Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|------|---------|
| Logistic Regression | 0.8771 | 0.8863 | 0.8771 | 0.8764 | 0.9501 |
| Random Forest | 0.9875 | 0.9876 | 0.9875 | 0.9875 | 0.9987 |
| XGBoost | 0.9882 | 0.9883 | 0.9882 | 0.9882 | 0.9986 |
| LightGBM | 0.9881 | 0.9881 | 0.9881 | 0.9881 | 0.9986 |

---

## Best Model: XGBoost

## Final Test Metrics

- **accuracy**: 0.9853
- **precision**: 0.9803
- **recall**: 0.9870
- **f1_score**: 0.9835
- **roc_auc**: 0.9988
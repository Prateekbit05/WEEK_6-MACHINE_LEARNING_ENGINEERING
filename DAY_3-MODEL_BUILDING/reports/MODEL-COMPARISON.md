# Model Comparison Report
## Day 3 - ML Engineering Week

**Generated:** 2026-03-12 18:41:26

---

## Cross-Validation Results (5-Fold)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Time |
|-------|----------|-----------|--------|----------|---------|------|
| LightGBM | 0.3172 | 0.2554 | 0.3172 | 0.2343 | 0.4993 | 85.8s |
| XGBoost | 0.3190 | 0.2528 | 0.3190 | 0.2262 | 0.5004 | 88.7s |
| Random Forest | 0.1420 | 0.2511 | 0.1420 | 0.1784 | 0.5000 | 21.4s |
| Neural Network | 0.3285 | 0.2385 | 0.3285 | 0.1779 | 0.4991 | 34.8s |
| Logistic Regression | 0.0073 | 0.2611 | 0.0073 | 0.0105 | 0.4996 | 159.6s |

---

## Best Model: LightGBM

### Test Set Metrics

- **Accuracy**: 0.3257
- **Precision**: 0.2291
- **Recall**: 0.3257
- **F1 Score**: 0.1835
- **Roc Auc**: 0.5002

---

## Output Files

| File | Description |
|------|-------------|
| `src/models/best_model.pkl` | Trained model |
| `src/evaluation/metrics.json` | All metrics |
| `plots/model_comparison.png` | Comparison chart |
| `plots/cm_lightgbm.png` | Confusion matrix |

---

## Day 3 Complete!
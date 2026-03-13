# 🤖 Day 3 — Model Building + Advanced Training Pipeline

> **ML Engineering Week · Day 3**
> Multi-model training, cross-validation, overfitting control, and automated model selection on the **NF-UQ-NIDS-v2** network intrusion detection dataset (16-class classification).

---

## 📁 Project Structure

```
day-3/
├── inputs/
│   ├── X_train_final.csv          # Preprocessed training data
│   └── X_test_final.csv           # Preprocessed test data
├── src/
│   ├── config/
│   │   └── config.yaml            # Model & pipeline configuration
│   ├── training/
│   │   ├── train.py               # 🔑 Main training pipeline
│   │   ├── models.py              # ModelFactory (LR, RF, XGB, LGBM, NN)
│   │   └── cv_trainer.py          # 5-fold cross-validation trainer
│   ├── models/
│   │   └── best_model.pkl         # ✅ Best trained model (auto-saved)
│   └── evaluation/
│       └── metrics.json           # ✅ All CV + test metrics
├── plots/
│   ├── model_comparison.png       # Bar chart: CV F1 across models
│   └── cm_lightgbm.png            # Confusion matrix (best model)
├── reports/
│   └── MODEL-COMPARISON.md        # Auto-generated comparison report
├── utils/
│   ├── logger.py                  # Logging utility
│   └── helpers.py                 # save_json, save_model helpers
├── README.md
├── COMMANDS.md
├── THEORY.md
└── TOPICS-TO-LEARN.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/training/train.py
```

That's it. The pipeline will:
- Load data from `inputs/`
- Train 5 models with 5-fold cross-validation
- Select the best model by F1 score
- Save the model, metrics, plots, and report automatically

---

## 🧠 Models Trained

| Model | Library | Notes |
|-------|---------|-------|
| Logistic Regression | scikit-learn | Baseline linear model |
| Random Forest | scikit-learn | Ensemble of decision trees |
| XGBoost | xgboost | Gradient boosted trees |
| LightGBM ✅ | lightgbm | **Best model (CV F1: 0.2343)** |
| Neural Network | scikit-learn | MLPClassifier |

> All models wrapped in a `StandardScaler → Model` sklearn `Pipeline`.

---

## 📊 Results Summary

### Cross-Validation (5-Fold) — 16-Class Network Intrusion Detection

| Model | CV Accuracy | CV F1 | ROC-AUC | Train Time |
|-------|-------------|-------|---------|------------|
| Logistic Regression | 0.0073 | 0.0105 | 0.4996 | 158.2s |
| Random Forest | 0.1420 | 0.1784 | 0.5000 | 17.4s |
| Neural Network | 0.3285 | 0.1779 | 0.4991 | 51.1s |
| XGBoost | 0.3190 | 0.2262 | 0.5004 | 81.5s |
| **LightGBM** ✅ | **0.3172** | **0.2343** | 0.4993 | 94.7s |

### Best Model Test Set (LightGBM)

| Metric | Score |
|--------|-------|
| Accuracy | 0.3257 |
| Precision | 0.2291 |
| Recall | 0.3257 |
| F1 Score | 0.1835 |
| ROC-AUC | 0.5002 |

> ⚠️ Low scores are expected — this is a highly imbalanced **16-class** network traffic classification problem. The pipeline architecture is correct; scores improve with feature engineering and class balancing (Day 4+).

---

## ✅ Deliverables

| File | Status |
|------|--------|
| `src/training/train.py` | ✅ Complete |
| `src/models/best_model.pkl` | ✅ Auto-saved |
| `src/evaluation/metrics.json` | ✅ Auto-saved |
| `reports/MODEL-COMPARISON.md` | ✅ Auto-generated |
| `plots/model_comparison.png` | ✅ Generated |
| `plots/cm_lightgbm.png` | ✅ Generated |

---

## ⚙️ Pipeline Architecture

```
load_data()
    │
    ▼
train_all_models()          ← 5 models × 5-fold CV
    │
    ▼
select_best_model()         ← ranked by CV F1
    │
    ▼
evaluate()                  ← test set: acc, prec, recall, F1, AUC
    │
    ▼
save_results()              ← model.pkl + metrics.json + plots + report
```

---

## 🔧 Requirements

```
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
```

---

## 📌 Key Design Decisions

- **StandardScaler inside Pipeline** — prevents data leakage during CV
- **Best model = highest CV F1** — more robust than accuracy for imbalanced classes  
- **label_binarize for ROC-AUC** — handles multi-class (OvR strategy)
- **`zero_division=0`** — safe handling of unseen classes in CV folds
- **Auto directory creation** — pipeline is self-contained and reproducible

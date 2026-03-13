# 🎯 TOPICS-TO-LEARN.md — Day 3 Deep Dives

Curated learning path based on what you built today. Each topic includes *why it matters* in the context of your pipeline.

---

## ✅ Covered Today (Solidify These)

### 1. sklearn Pipeline & Data Leakage
**What you did:** Wrapped every model in `Pipeline([scaler, model])`  
**Go deeper:**
- [ ] Read: [sklearn Pipeline docs](https://scikit-learn.org/stable/modules/pipeline.html)
- [ ] Practice: Deliberately introduce leakage, observe inflated scores, fix it
- [ ] Explore: `ColumnTransformer` for mixed feature types (numeric + categorical)

---

### 2. 5-Fold Cross-Validation
**What you did:** `CrossValidationTrainer(n_splits=5)` on all models  
**Go deeper:**
- [ ] Understand the bias-variance tradeoff of k (try k=3, k=10, compare)
- [ ] Learn `StratifiedKFold` — preserves class ratios in each fold (critical for your 16-class dataset)
- [ ] Learn `TimeSeriesSplit` — for sequential/temporal data

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

### 3. Evaluation Metrics for Imbalanced Classes
**What you saw:** Neural Net had highest accuracy (0.3285) but worse F1 (0.1779) than LightGBM  
**Go deeper:**
- [ ] Understand macro vs micro vs weighted averaging
- [ ] Learn `classification_report` for per-class breakdown
- [ ] Study `Cohen's Kappa` — another imbalance-robust metric
- [ ] Practice: compute all 5 metrics manually on a small dataset

---

### 4. LightGBM vs XGBoost
**What you saw:** LightGBM won (F1: 0.2343 vs 0.2262) and was more memory efficient  
**Go deeper:**
- [ ] Read LightGBM's [leaf-wise growth paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- [ ] Understand GOSS and EFB in detail
- [ ] Compare `num_leaves` (LightGBM) vs `max_depth` (XGBoost)
- [ ] Practice: tune `learning_rate` × `n_estimators` tradeoff

---

## 🔶 Next Priority Topics

### 5. Handling Class Imbalance
**Why it matters for you:** Your 16-class NIDS data has extreme imbalance — rare attack types get poor recall  
**Topics:**
- [ ] **SMOTE** (Synthetic Minority Oversampling) — `imbalanced-learn` library
- [ ] **Class weights** — `class_weight='balanced'` in sklearn models
- [ ] **Undersampling** — RandomUnderSampler
- [ ] **Focal Loss** — used in neural nets for imbalanced classification

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
```

---

### 6. Hyperparameter Tuning
**Why it matters:** Your models ran on default params — tuning could significantly improve F1  
**Topics:**
- [ ] `GridSearchCV` — exhaustive search (slow but thorough)
- [ ] `RandomizedSearchCV` — faster, good for large spaces
- [ ] `Optuna` — modern Bayesian optimization (best for LightGBM/XGBoost)
- [ ] `HalvingGridSearchCV` — successive halving for speed

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000)
    }
    model = LGBMClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

### 7. Feature Importance & Selection
**Why it matters:** LightGBM and XGBoost rank features — you can remove noise features to speed up training and improve generalization  
**Topics:**
- [ ] `model.feature_importances_` in tree models
- [ ] SHAP values — model-agnostic, shows *direction* of feature effect
- [ ] `SelectFromModel` — automatic feature selection based on importance threshold
- [ ] Permutation importance — more reliable than split-based importance

```python
import shap
explainer = shap.TreeExplainer(model['model'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

---

### 8. Regularization Deep Dive
**Why it matters:** Understanding L1/L2 helps tune XGBoost/LightGBM and Logistic Regression  
**Topics:**
- [ ] Visualize L1 vs L2 penalty geometry (diamond vs circle constraint)
- [ ] Ridge regression from scratch (closed-form solution exists)
- [ ] Lasso path — how features enter the model as λ decreases
- [ ] `ElasticNet` — combined penalty, used in `SGDClassifier`
- [ ] Dropout in neural networks — equivalent to L2 regularization (Srivastava 2014)

---

## 🔷 Advanced Topics (Week 2+)

### 9. Gradient Boosting Theory
- [ ] Understand the **gradient descent in function space** intuition
- [ ] Derive why log-loss gradient = residuals for classification
- [ ] Read: Chen & Guestrin XGBoost paper (2016)
- [ ] Learn about `shrinkage` (learning rate) and why smaller = better with more trees

### 10. Neural Network Architectures for Tabular Data
- [ ] **TabNet** — attention-based feature selection
- [ ] **NODE** — Neural Oblivious Decision Ensembles
- [ ] **FT-Transformer** — transformers for tabular data
- [ ] When do NNs actually beat GBMs on tabular data? (Spoiler: rarely, but it happens)

### 11. Multi-Label vs Multi-Class Classification
- [ ] Understand the difference (your problem is **multi-class**, not multi-label)
- [ ] One-vs-Rest (OvR) strategy
- [ ] Classifier chains for multi-label
- [ ] How `label_binarize` works internally

### 12. Model Calibration
- [ ] What calibration means (predicted probabilities = actual frequencies)
- [ ] Reliability diagrams (calibration curves)
- [ ] `CalibratedClassifierCV` — Platt scaling and isotonic regression
- [ ] Why this matters for ROC-AUC interpretation

---

## 📖 Recommended Reading

| Resource | Topic | Priority |
|----------|-------|----------|
| [ESL Chapter 10](https://hastie.su.domains/ElemStatLearn/) | Boosting theory | ⭐⭐⭐ |
| [LightGBM paper](https://papers.nips.cc/paper/6907) | GOSS + EFB | ⭐⭐⭐ |
| [SHAP paper (Lundberg 2017)](https://arxiv.org/abs/1705.07874) | Feature explanability | ⭐⭐⭐ |
| [Imbalanced-learn docs](https://imbalanced-learn.org/) | Class imbalance | ⭐⭐⭐ |
| [Optuna docs](https://optuna.readthedocs.io/) | Hyperparameter tuning | ⭐⭐ |
| [sklearn model selection guide](https://scikit-learn.org/stable/model_selection.html) | CV strategies | ⭐⭐ |

---

## 🔍 Questions to Answer Before Day 4

1. Why did Logistic Regression get ~0% F1 on your dataset specifically?
2. Why is Neural Network accuracy high but F1 low — what does that tell you about class distribution?
3. What would `StratifiedKFold` change vs regular `KFold` in your CV results?
4. If you added `class_weight='balanced'` to all models, which would benefit most?
5. Your ROC-AUC ≈ 0.500 across all models — what does this indicate about class separability in feature space?

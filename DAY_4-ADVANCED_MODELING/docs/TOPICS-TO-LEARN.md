# 🎯 TOPICS-TO-LEARN.md — Day 4 Deep Dives

Curated learning path based on what you built today — tuning, explainability, and error analysis.

---

## ✅ Covered Today (Solidify These)

### 1. Optuna — Bayesian Hyperparameter Optimization
**What you did:** TPE sampler, 30 trials, `direction='maximize'`  
**Go deeper:**
- [ ] Understand **TPE vs CMA-ES vs Random** sampler tradeoffs
- [ ] Learn Optuna **pruning** — stop bad trials early with `SuccessiveHalvingPruner`
- [ ] Use `optuna.visualization` for optimization history plots
- [ ] Persist studies with `optuna.create_study(storage='sqlite:///study.db')` — resume across sessions

```python
# Pruning example — stops bad trials early
import optuna
from optuna.pruners import MedianPruner

study = optuna.create_study(pruner=MedianPruner())
```

---

### 2. SHAP TreeExplainer
**What you did:** Computed exact SHAP values, saved summary + bar plots  
**Go deeper:**
- [ ] **Waterfall plot** — explain a single prediction step-by-step
- [ ] **Dependence plot** — show how one feature's SHAP value changes with its magnitude
- [ ] **Force plot** — interactive single-prediction explanation
- [ ] **KernelExplainer** — model-agnostic SHAP for non-tree models (slower but universal)

```python
# Waterfall plot — explain one prediction
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_sample.iloc[0],
    feature_names=X_sample.columns.tolist()
))

# Dependence plot — how feature 12 interacts with feature 11
shap.dependence_plot("Feature_12", shap_values, X_sample, interaction_index="Feature_11")
```

---

### 3. Feature Importance — Multiple Methods
**What you saw:** SHAP mean absolute importance ranked features 12, 11, 3 as top  
**Go deeper:**
- [ ] Compare **3 importance methods** on same model: split-based vs SHAP vs permutation
- [ ] Learn why split-based importance is biased toward high-cardinality features
- [ ] **Permutation importance** — model-agnostic, measures actual prediction degradation

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10, scoring='f1_macro')
# Compare with SHAP ranking — they should broadly agree
```

---

## 🔶 Next Priority Topics

### 4. Advanced Optuna Patterns
**Why it matters:** 30 trials found good params but not necessarily the global optimum  
**Topics:**
- [ ] **Multi-objective optimization** — optimize F1 AND inference time simultaneously
- [ ] **Optuna + LightGBM integration** — `LightGBMTuner` (built-in, uses pruning automatically)
- [ ] **Search space design** — when to use `suggest_int` vs `suggest_float` vs `suggest_categorical`
- [ ] **Warm starting** — initialize Optuna with Day 3 best params

```python
import optuna.integration.lightgbm as lgb_optuna

# LightGBM has native Optuna integration
tuner = lgb_optuna.LightGBMTuner(params, dtrain, valid_sets=[dvalid])
tuner.run()
print("Best params:", tuner.best_params)
```

---

### 5. Error Analysis & Confusion Patterns
**Why it matters:** Your model scores 0.9850 F1 — what does the remaining 1.5% get wrong?  
**Topics:**
- [ ] **Per-class precision/recall** — `classification_report` to find weakest attack classes
- [ ] **Error clustering** — group misclassified samples by feature similarity (t-SNE or UMAP)
- [ ] **Confusion heatmap** — which attack classes get confused with each other?
- [ ] **Hard example mining** — find samples the model is consistently wrong on across CV folds

```python
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Per-class breakdown
print(classification_report(y_test, y_pred, digits=4))

# Visual confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', cmap='Blues')
plt.title('Normalized Confusion Matrix')
plt.savefig('plots/confusion_normalized.png', dpi=150)
```

---

### 6. Model Calibration
**Why it matters:** `predict_proba()` output — are those probabilities trustworthy?  
**Topics:**
- [ ] **Reliability diagram** — plot predicted probability vs actual frequency per bin
- [ ] **Brier score** — measures calibration quality (lower = better calibrated)
- [ ] **Platt scaling** — post-hoc sigmoid calibration
- [ ] **Isotonic regression** — non-parametric calibration (better for RF)

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Check calibration
prob_true, prob_pred = calibration_curve(y_test_binary, y_proba[:, 1], n_bins=10)

# Fix calibration
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)
```

---

### 7. Global vs Local Explainability
**Why it matters:** SHAP gives both — you used global today, local is powerful for debugging  
**Topics:**
- [ ] **Global** = average behaviour across all predictions (what you built)
- [ ] **Local** = why was *this specific packet* classified as an attack?
- [ ] **LIME** (Local Interpretable Model-agnostic Explanations) — alternative to SHAP for local explanations
- [ ] **Anchors** — rule-based local explanations ("IF feature_12 > 0.5 AND feature_3 < 0.2 THEN attack")

```python
# Local SHAP explanation for the hardest misclassified sample
wrong_idx = np.where(y_pred != y_test.values)[0][0]
print(f"Predicted: {y_pred[wrong_idx]}, Actual: {y_test.values[wrong_idx]}")
shap.plots.waterfall(shap.Explanation(
    values=shap_values[wrong_idx],
    base_values=explainer.expected_value[1],
    data=X_sample.iloc[wrong_idx]
))
```

---

## 🔷 Advanced Topics (Week 2+)

### 8. AutoML — Automated Pipeline Search
- [ ] **TPOT** — evolves entire sklearn pipelines using genetic algorithms
- [ ] **Auto-sklearn** — meta-learning + Bayesian optimization over 100+ sklearn models
- [ ] **H2O AutoML** — industrial-strength, handles preprocessing automatically
- [ ] When to use AutoML vs manual tuning in production

### 9. MLflow — Experiment Tracking
- [ ] Log every Optuna trial to MLflow for reproducibility
- [ ] Compare runs visually in the MLflow UI
- [ ] Register the best model to MLflow Model Registry
- [ ] `mlflow.sklearn.autolog()` — zero-code logging for sklearn

```python
import mlflow

mlflow.set_experiment("day4-tuning")
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.sklearn.log_model(best_model, "model")
```

### 10. Production Model Serving
- [ ] **ONNX** — convert sklearn model to universal format for fast inference
- [ ] **BentoML** — package model as REST API in one command
- [ ] **FastAPI + joblib** — serve `best_random_forest.joblib` as an endpoint
- [ ] Latency benchmarking — measure inference time per sample

### 11. Responsible AI Checks
- [ ] **Fairness audit** — does the model perform equally across network types/sources?
- [ ] **Adversarial robustness** — can an attacker craft packets to fool the model?
- [ ] **Data drift detection** — how to detect when production traffic distribution shifts
- [ ] **Explainability for security teams** — translating SHAP values into actionable network rules

---

## 📖 Recommended Reading

| Resource | Topic | Priority |
|----------|-------|----------|
| [Optuna paper (Akiba 2019)](https://arxiv.org/abs/1907.10902) | TPE + pruning theory | ⭐⭐⭐ |
| [SHAP paper (Lundberg 2017)](https://arxiv.org/abs/1705.07874) | Shapley values for ML | ⭐⭐⭐ |
| [Bergstra & Bengio 2012](https://jmlr.org/papers/v13/bergstra12a.html) | Random > Grid search | ⭐⭐⭐ |
| [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/) | Full explainability guide | ⭐⭐⭐ |
| [MLflow docs](https://mlflow.org/docs/latest/) | Experiment tracking | ⭐⭐ |
| [sklearn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) | Probability calibration | ⭐⭐ |

---

## 🔍 Questions to Answer Before Day 5

1. Your top feature is Feature 12 (SHAP = 0.1132) — what does this index correspond to in the original NF-UQ-NIDS-v2 column names? Does it make sense for network intrusion detection?
2. Why did Optuna find `n_estimators=292` rather than a round number like 300? What does this tell you about the search?
3. The tuned model scored lower F1 than baseline — under what conditions would you still deploy the tuned model over the baseline?
4. Your SHAP analysis used only 100 samples — if you used 10,000 samples, how would the importance rankings likely change?
5. Features 11 and 12 have very similar SHAP importance (0.1132 vs 0.1014) — could they be correlated? How would you check?

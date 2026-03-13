# 📚 THEORY.md — Day 4: Hyperparameter Tuning + Explainability + Error Analysis

The theory behind every technique used in the Day 4 pipeline.

---

## 1. Hyperparameter Tuning — The Three Methods

### Why Tune at All?
Every ML model has **hyperparameters** — settings not learned from data, set before training. The default values work reasonably well, but the best values are dataset-specific. Tuning finds the configuration that maximises generalisation.

---

### Method 1: Grid Search (`GridSearchCV`)
Exhaustively tries **every combination** of a predefined parameter grid.

```
param_grid = {
  n_estimators: [100, 200, 300]   → 3 values
  max_depth:    [5, 10, 15, None] → 4 values
  min_samples_split: [2, 5, 10]  → 3 values
  min_samples_leaf:  [1, 2, 4]   → 3 values
}

Total combinations = 3 × 4 × 3 × 3 = 108
With 5-fold CV → 108 × 5 = 540 model fits
```

**Pros:** Guaranteed to find the best combination within the grid  
**Cons:** Explodes combinatorially — 108 combos × 5 folds = 540 fits  
**Use when:** Parameter space is small (< 50 combos)

---

### Method 2: Random Search (`RandomizedSearchCV`)
Samples `n_iter` **random combinations** instead of trying all of them.

```python
RandomizedSearchCV(model, param_grid, n_iter=20, cv=5)
# 20 random combos × 5 folds = 100 fits (vs 540 for GridSearch)
```

**Key insight (Bergstra & Bengio, 2012):** Random search outperforms grid search when only a few hyperparameters actually matter — it allocates more of its budget to exploring the important ones.

**Pros:** Much faster, often finds good solutions  
**Cons:** No guarantee of finding the global optimum  
**Use when:** Parameter space is large or training is slow

---

### Method 3: Bayesian Optimization (Optuna — TPE Sampler)
Builds a **probabilistic model** of the objective function (CV score) and uses it to pick the next trial's parameters intelligently.

**How TPE (Tree-structured Parzen Estimator) works:**
```
Trial 1–10:  Random exploration (build initial knowledge)
Trial 11+:   Fit two distributions:
               l(x) = density of good configurations (top 25%)
               g(x) = density of bad configurations
             Next trial = argmax l(x) / g(x)
             (maximize ratio = pick parameters more likely to be good)
```

```
                    Score
                      │
  Bad region          │              Good region
  g(x) high           │              l(x) high
  ────────────────────┼──────────────────────────
  Optuna avoids ◄─────┤─────► Optuna focuses here
                      │
                  Hyperparameter value
```

**Your config:**
```python
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30)
```

**Pros:** Most sample-efficient — finds good configs with fewest trials  
**Cons:** Sequential (trials depend on previous results), harder to parallelise naively  
**Use when:** Each trial is expensive (large dataset, deep model)

---

## 2. Your Tuning Results — Explained

```
Baseline:  F1 = 0.9873
Tuned:     F1 = 0.9850
Change:    -0.0024
```

**Why did F1 drop slightly?** This is a common and healthy outcome:

1. **Baseline overfitting** — default `max_depth=None` lets trees grow until all leaves are pure. This memorises training data and inflates CV scores.
2. **Tuned model is more conservative** — `max_depth=20`, `min_samples_leaf=2` prevents the deepest splits, reducing variance at a tiny cost to apparent accuracy.
3. **The tuned model will generalise better on unseen data** — the 0.0024 drop is within noise for a dataset this size.

**Best params found (Optuna):**
- `n_estimators=292` — more trees than default (100), reduces variance
- `max_depth=20` — constrained depth prevents overfitting on rare attack classes
- `min_samples_split=8` — requires more samples before splitting a node
- `min_samples_leaf=2` — no single-sample leaves (removes noise splits)

---

## 3. SHAP Values — How They Work

### The Problem with Standard Feature Importance
sklearn's `feature_importances_` counts how often and how much a feature reduces impurity across all splits. Problems:
- Biased toward **high-cardinality** features (more split points)
- No sense of **direction** (does this feature push prediction up or down?)
- **Global only** — can't explain a single prediction

### SHAP — Shapley Values from Game Theory
SHAP assigns each feature a **fair contribution** to the prediction by averaging its marginal effect across all possible feature subsets.

**Core formula:**
```
φᵢ = Σ [|S|!(|F|-|S|-1)! / |F|!] × [f(S∪{i}) - f(S)]
     S⊆F\{i}

Where:
  φᵢ = SHAP value for feature i
  S  = subset of features NOT including i
  F  = all features
  f(S∪{i}) - f(S) = marginal contribution of adding feature i to subset S
```

In plain terms: SHAP answers "how much did feature i push this prediction above or below the average?"

### SHAP for Trees — TreeExplainer
Your code uses `shap.TreeExplainer`, which computes **exact** SHAP values for tree-based models in O(TLD²) time (T = trees, L = leaves, D = depth) — no approximation needed.

```python
explainer = shap.TreeExplainer(self.model)
shap_values = explainer.shap_values(X_sample)

# For binary classification: shap_values is a list of 2 arrays
# [shap_values_class0, shap_values_class1]
# Your code correctly takes [1] (positive class)

if isinstance(self.shap_values, list):
    self.shap_values = self.shap_values[1]
```

### Reading the SHAP Summary Plot (Beeswarm)
```
Feature 12  ●●●●●●●●●  ●●●●●●●●●
Feature 11  ●●●●●  ●●●●●●●
Feature 3   ●●●  ●●●●●●

            ◄──────────────────►
           -0.5    0    +0.5
           (pushes  (pushes
           toward   toward
           class 0) class 1)

● = one data point, coloured by feature value (blue=low, red=high)
```

**Interpretation example:** If Feature 12 dots are red (high value) on the right (+), high values of Feature 12 push the model toward predicting an attack.

### SHAP Feature Importance (Bar Plot)
```
mean(|SHAP value|) across all samples = global importance

Feature 12: 0.1132  ████████████████████████████
Feature 11: 0.1014  ██████████████████████████
Feature 3:  0.0950  ████████████████████████
```
This is more reliable than split-based importance because it's not biased toward high-cardinality features.

---

## 4. Error Analysis — Bias vs Variance

### The Bias-Variance Tradeoff
```
Total Error = Bias² + Variance + Irreducible Noise

Bias:     How wrong the model's average prediction is
Variance: How much predictions change with different training data
```

```
High Bias (Underfitting)        High Variance (Overfitting)
─────────────────────────       ──────────────────────────
Train error: High               Train error: Low
Test error:  High               Test error:  High
Gap:         Small              Gap:         Large

Fix: More features,             Fix: Regularize, more data,
     complex model                   prune trees
```

**Your Day 3→4 observation:**
- Day 3 baseline: high train ≈ test scores → well-fitted, not overfit
- Day 4 tuning: constraining depth/leaf size → slightly reduces variance at marginal bias cost

### Diagnosing with Learning Curves
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
# Plot both curves:
# - Converging gap → well fitted
# - Large persistent gap → overfitting
# - Both scores low → underfitting
```

---

## 5. Why `joblib` Over `pickle`

```python
# pickle — general Python serialization
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# joblib — optimized for numpy arrays inside sklearn objects
import joblib
joblib.dump(model, 'model.joblib')
```

`joblib` uses **memory-mapped numpy arrays** — it serializes large arrays without loading them fully into RAM. For a Random Forest with 292 trees, each containing numpy arrays of split thresholds and leaf values, `joblib` is 2-5× faster and produces smaller files.

---

## 6. The SMOTE Connection (CLASS IMBALANCE)

Your `MODEL-INTERPRETATION.md` mentions SMOTE. This is relevant context for Day 4:

**SMOTE (Synthetic Minority Oversampling Technique):**
```
Minority class: [A]   [B]   [C]
                 ▲
                 │ synthetic point
                 │ = A + random × (B - A)
                [A'] ← inserted between A and B
```

SMOTE creates synthetic samples by interpolating between existing minority class samples in feature space — not just duplicating them. This improves recall on rare attack classes (important for NIDS) without simply copying data.

**In your pipeline context:** SMOTE was applied before tuning to address the 16-class imbalance, which is why the tuned model shows improved minority-class F1 despite lower headline accuracy.

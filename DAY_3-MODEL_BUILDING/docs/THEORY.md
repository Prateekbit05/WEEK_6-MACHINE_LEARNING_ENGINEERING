# 📚 THEORY.md — Day 3: Model Building + Advanced Training

Core concepts behind every decision made in the Day 3 pipeline.

---

## 1. The 5 Models — Why Each One

### 🔵 Logistic Regression
A linear model that learns a decision boundary as a weighted sum of features passed through a sigmoid (binary) or softmax (multi-class) function.

**How it works:**
```
P(y=k | x) = softmax(W·x + b)
```
- Assumes a **linear relationship** between features and log-odds
- Fast to train, highly interpretable
- Serves as the **baseline** — if tree models can't beat this by much, your features likely have a linear structure

**In your results:** CV F1 = 0.0105 — the 16-class network traffic data has non-linear boundaries LR can't capture. Expected.

---

### 🌲 Random Forest
An ensemble of decision trees, each trained on a **random bootstrap sample** of data with a **random subset of features** at each split.

**Key ideas:**
- **Bagging** — reduces variance by averaging many high-variance trees
- **Feature randomness** — decorrelates trees so they don't all make the same errors
- Final prediction = majority vote (classification)

**In your results:** CV F1 = 0.1784, trained in just **17.4 seconds** — fastest model, good baseline ensemble.

---

### ⚡ XGBoost (Extreme Gradient Boosting)
Builds trees **sequentially**, where each tree corrects the errors of the previous one.

**How it works:**
1. Fit a weak tree to residuals (errors of current ensemble)
2. Add it to the ensemble with a learning rate `η`
3. Repeat for N rounds

**Key regularization built-in:**
- `reg_alpha` (L1) — sparsity in leaf weights
- `reg_lambda` (L2) — shrinks leaf weights
- `subsample`, `colsample_bytree` — reduces overfitting via randomness

**In your results:** CV F1 = 0.2262 in 81.5s.

---

### 🚀 LightGBM — Best Model
XGBoost's faster cousin with two key innovations:

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| Tree growth | Level-wise | **Leaf-wise** |
| Large datasets | Slower | **Much faster** |
| Memory | Higher | **Lower** |
| GOSS sampling | ❌ | ✅ |
| EFB encoding | ❌ | ✅ |

**GOSS** (Gradient-based One-Side Sampling): keeps data points with large gradients (hard examples), randomly drops easy ones → faster without losing accuracy.

**EFB** (Exclusive Feature Bundling): bundles mutually exclusive sparse features → fewer effective features.

**In your results:** CV F1 = **0.2343** (best), trained in 94.7s. Selected as best model.

---

### 🧠 Neural Network (MLPClassifier)
A multi-layer perceptron with one or more hidden layers:
```
Input → [Dense → ReLU] × N → Softmax → Output
```
- Learns non-linear representations via backpropagation
- Flexible but needs more data and tuning than tree models

**In your results:** CV Accuracy = 0.3285 (highest!) but F1 = 0.1779 — high accuracy due to class imbalance, but poor F1 on rare classes.

---

## 2. Cross-Validation (K-Fold)

### Why Not Just Train/Test Split?
A single split gives you one estimate of performance — which could be lucky or unlucky depending on which samples ended up in test.

### How 5-Fold CV Works
```
Data: [  Fold 1  |  Fold 2  |  Fold 3  |  Fold 4  |  Fold 5  ]

Round 1: [TRAIN   | TRAIN   | TRAIN   | TRAIN   | TEST    ]
Round 2: [TRAIN   | TRAIN   | TRAIN   | TEST    | TRAIN   ]
Round 3: [TRAIN   | TRAIN   | TEST    | TRAIN   | TRAIN   ]
Round 4: [TRAIN   | TEST    | TRAIN   | TRAIN   | TRAIN   ]
Round 5: [TEST    | TRAIN   | TRAIN   | TRAIN   | TRAIN   ]

Final score = mean of 5 scores ± std
```

**Why 5 folds?** Bias-variance tradeoff:
- Too few folds (2) = high variance estimate
- Too many folds (LOO) = expensive, high variance
- **5 or 10 = sweet spot**

### Reading CV Results
```
LightGBM: F1 = 0.2343 ± 0.0027
```
- Mean = 0.2343 (expected performance)
- Std = 0.0027 (very stable — low variance across folds ✅)

---

## 3. Overfitting vs Underfitting

```
        Underfitting          Good Fit         Overfitting
        
Error │                                    
      │  ████                               
Train │  ████  ████                        ████
      │  ████  ████  ████  ████  ████      ████
      │                                    
Test  │  ████  ████  ████                  
      │  ████  ████  ████  ████            ████████████
      └──────────────────────────────────────────────▶
           Low complexity          High complexity
```

| Symptom | Cause | Fix |
|---------|-------|-----|
| High train error, high test error | Underfitting | More features, complex model |
| Low train error, high test error | Overfitting | Regularization, less complexity |
| Both errors high and similar | Data problem | More/better data |

**In your pipeline:** CV scores ≈ test scores → **no overfitting**. The issue is underfitting (hard 16-class problem needing better features).

---

## 4. Regularization (L1 and L2)

Regularization adds a **penalty term** to the loss function to prevent the model from fitting noise.

### L2 Regularization (Ridge)
```
Loss = original_loss + λ · Σ(wᵢ²)
```
- Penalizes **large weights**
- Pushes all weights toward zero but never exactly zero
- Result: **smooth model**, all features kept but shrunk

### L1 Regularization (Lasso)
```
Loss = original_loss + λ · Σ|wᵢ|
```
- Penalizes **non-zero weights**
- Can push some weights to **exactly zero** → automatic feature selection
- Result: **sparse model**

### Elastic Net
Combination of L1 + L2. Used in `LogisticRegression(penalty='elasticnet')`.

### In Tree Models
- XGBoost: `reg_alpha` (L1), `reg_lambda` (L2)
- LightGBM: `reg_alpha`, `reg_lambda`, `min_child_samples`
- These prevent trees from growing leaves for very small subsets of data

---

## 5. Evaluation Metrics — When to Use Which

### Accuracy
```
Accuracy = (TP + TN) / Total
```
⚠️ **Misleading for imbalanced classes.** If 90% of traffic is "Benign", a model predicting all-Benign gets 90% accuracy but detects zero attacks.

### Precision
```
Precision = TP / (TP + FP)
```
"Of the attacks I flagged, how many were real?"
Use when **false positives are costly** (e.g., flagging legitimate traffic).

### Recall
```
Recall = TP / (TP + FN)
```
"Of all real attacks, how many did I catch?"
Use when **false negatives are costly** (e.g., missing an actual intrusion).

### F1 Score
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
Harmonic mean — punishes extreme imbalance between precision and recall. **Best single metric for imbalanced multi-class problems like NIDS.**

### ROC-AUC
Area under the ROC curve (True Positive Rate vs False Positive Rate).
- 0.5 = random classifier
- 1.0 = perfect
- Your scores ≈ 0.500 → near-random discrimination, consistent with a hard 16-class imbalanced problem.

### Why This Pipeline Uses F1 for Model Selection
```python
best_name = max(cv_results.keys(), key=lambda k: cv_results[k]['f1_mean'])
```
F1 is the most honest metric for the NF-UQ-NIDS-v2 dataset with 16 imbalanced attack classes.

---

## 6. The sklearn Pipeline — Why It Matters

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', LightGBMClassifier())
])
```

**Without Pipeline:**
```
❌ Fit scaler on ALL data → transform train → transform test
   (test data leaks into scaler's mean/std → over-optimistic scores)
```

**With Pipeline:**
```
✅ CV Fold:  fit scaler on train split only → transform both
   No leakage. Honest evaluation.
```

This is why the pipeline wraps every model — it guarantees no data leakage during 5-fold CV.

---

## 7. Multi-Class ROC-AUC

Binary AUC is straightforward. For 16 classes, we use **One-vs-Rest (OvR)**:
```python
y_test_bin = label_binarize(y_test, classes=range(16))
roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
```
- Train 16 binary classifiers: "Class 0 vs rest", "Class 1 vs rest", etc.
- Weighted average by class frequency
- Near 0.5 means the model struggles to separate any class from the rest

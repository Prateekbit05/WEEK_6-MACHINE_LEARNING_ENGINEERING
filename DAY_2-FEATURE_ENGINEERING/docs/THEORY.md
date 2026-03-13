# 📚 THEORY.md — Day 2: Feature Engineering + Feature Selection

---

## 1. What is Feature Engineering?

Feature engineering is the process of using domain knowledge and mathematical transformations to create new input variables from existing ones. Raw features often don't expose the full pattern in the data — feature engineering makes hidden patterns explicit and easier for models to learn.

**Your pipeline went from 42 → 144 features across 8 transformation types.** After selection, 50 of those 144 were kept — meaning 50 engineered features were judged more informative than the 94 that were dropped, including some of the original raw features.

The core principle: **models learn from numbers, not meaning.** Feature engineering translates domain meaning into numbers the model can exploit.

---

## 2. Categorical Encoding

### Why it's needed
ML models only understand numbers. Categorical columns (text labels, categories) must be converted to integers before any model can use them.

Your dataset has 4 categorical columns (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `Attack`, `Label`). The `Attack` target column is encoded separately using `LabelEncoder`.

### Strategy 1 — One-Hot Encoding (OHE)
Creates a new binary column for each category. Used when **cardinality is low** (≤ 10 unique values by default in your config).

```
PROTOCOL:  TCP → [1, 0, 0]
           UDP → [0, 1, 0]
           ICMP→ [0, 0, 1]
```

- **Pros:** No ordinal assumption, model treats each category independently
- **Cons:** Creates many columns for high-cardinality features (curse of dimensionality)
- **Your rule:** OHE if unique values ≤ `onehot.max_categories` from config

### Strategy 2 — Label Encoding
Converts each unique category to an integer (0, 1, 2, …). Used when **cardinality is high**.

```
PROTOCOL: TCP=0, UDP=1, ICMP=2, OTHER=3
```

- **Pros:** No column explosion, memory efficient
- **Cons:** Implies ordinal ordering (TCP < UDP < ICMP) which may mislead linear models
- **Safe for:** Tree-based models (Random Forest, XGBoost) which split on thresholds

### Handling Unseen Categories in Test Set
Your pipeline handles this correctly — if test has a value not seen in train, it maps to `'unknown'` before label encoding:
```python
X_test[col] = X_test[col].apply(
    lambda x: x if x in le.classes_ else 'unknown'
)
```
Without this, `transform()` would crash on new values.

### Target Encoding (concept — not used directly here)
Replaces each category with the mean target value for that category. Powerful but prone to leakage if not done carefully with cross-validation folds.

---

## 3. Numerical Transformations

### Why transform numerical features?

Many ML algorithms (especially linear ones) perform better when features are normally distributed and on similar scales. Highly skewed features like `IN_BYTES` or `OUT_BYTES` benefit significantly from transformation.

### Log Transformation — `feature_log`

```
new_feature = log(x + 1)
```

The `+1` handles zero values (log(0) = undefined).

**Use when:** Feature is right-skewed (long tail of large values), e.g. byte counts, packet sizes, flow durations.

**Effect:** Compresses large values, spreads small values. `IN_BYTES` of [1, 10, 100, 1000, 1000000] becomes [0, 2.3, 4.6, 6.9, 13.8] — much more manageable range.

**In your results:** `L4_SRC_PORT_log` scored highest MI (0.0347) among log features — the log transformation exposed port-based attack patterns that the raw value obscured.

### Sqrt Transformation — `feature_sqrt`

```
new_feature = sqrt(x)
```

**Use when:** Mild right skew, or when you want a gentler compression than log. Works on zero values directly.

**In your results:** `IN_BYTES_sqrt` scored 0.0257 MI — retained in final 50 features.

### Power Transformation — `feature_pow2`

```
new_feature = x ^ power    (e.g., x²)
```

**Use when:** Amplifying differences between feature values, especially for features where small differences at large values matter more. Opposite of log — spreads out large values further.

**In your results:** Only `PROTOCOL_pow2` was created — PROTOCOL has values 6 (TCP), 17 (UDP), 1 (ICMP). Squaring amplifies these differences: 36, 289, 1 — easier for linear models to separate.

---

## 4. Interaction and Ratio Features

### Interaction Features — `feature1_x_feature2`

```
new_feature = feature_A × feature_B
```

Captures combined effects that neither feature reveals individually.

**Example from your data:** `L4_SRC_PORT_x_PROTOCOL` (MI = 0.0365, highest overall)

A DDoS attack from port 80 using UDP is meaningfully different from normal HTTP traffic on port 80 using TCP. The raw features `L4_SRC_PORT` and `PROTOCOL` separately don't capture this — but their product does.

**Your pipeline:** Selects the top 6 features by variance, creates all pairwise products (up to 15 combinations).

### Ratio Features — `feature1_div_feature2`

```
new_feature = feature_A / (feature_B + epsilon)
```

The `epsilon` prevents division by zero.

**Example:** `IN_BYTES_div_IN_PKTS` = bytes per packet = average packet size.
- Normal HTTP: ~1400 bytes/packet
- DDoS SYN flood: ~60 bytes/packet (tiny packets, huge count)
- Data exfiltration: ~1500 bytes/packet (maximum size packets)

The ratio reveals the attack type in a way neither raw feature could alone. **MI score: 0.0284** — ranked in final 50.

---

## 5. Aggregation Features

Your `AggregationTransformer` creates row-level summary statistics across all numerical features:

```
agg_sum    = sum of all feature values in that row
agg_mean   = mean of all feature values
agg_std    = std deviation across features in that row
agg_max    = maximum feature value in the row
agg_min    = minimum feature value in the row
agg_median = median feature value in the row
agg_skew   = skewness across features in the row
```

**Why useful for network intrusion detection:**
- A benign flow has moderate, balanced feature values — low `agg_std`
- A DDoS flow has extreme values for a few features (bytes, packets) — high `agg_std`, high `agg_sum`
- An idle/scanning flow has mostly zero values — near-zero `agg_sum`

**In your results:** `agg_sum` scored 0.0335 MI and had the highest GB importance (0.210) — the row sum is the single strongest gradient boosting feature.

---

## 6. Binning Features — `feature_binned`

Converts a continuous numerical feature into discrete categories (bins).

```
Quantile binning (strategy='quantile'):
  Bin 0: values in 0-20th percentile
  Bin 1: values in 20-40th percentile
  Bin 2: values in 40-60th percentile
  Bin 3: values in 60-80th percentile
  Bin 4: values in 80-100th percentile
```

**Why quantile over equal-width:**
Equal-width bins on `OUT_BYTES` would put 99% of traffic in bin 0 (since most flows are small) and one extreme DDoS flow in bin 4. Quantile binning always has equal population per bin — much more informative.

**In your results:** `OUT_BYTES_binned` achieved the highest MI score overall (0.0372). Binning converted a highly skewed byte count into a clean ordinal category that perfectly stratifies traffic volume.

---

## 7. Statistical Features — Percentile Encoding

```
feature_percentile = rank of value / total count  (0 to 1)
```

Instead of the raw value, stores where this observation falls in the overall distribution.

**Example:** `IN_BYTES_percentile` = 0.95 means "this flow has more bytes than 95% of all flows" — immediately identifies large transfers regardless of the absolute byte count.

**In your results:** `IN_BYTES_percentile` scored 0.0257 MI and was retained in final 50 features.

---

## 8. Feature Selection — Why It Matters

After engineering, you have 144 features. More features is not always better:
- More features = longer training time
- Irrelevant features add noise, hurting model accuracy
- Redundant features cause multicollinearity, destabilising linear models
- Overfitting risk increases with feature count relative to sample count

Your pipeline reduces 144 → 50 using 6 methods — removing 94 features (65%).

---

## 9. Variance Threshold

```python
VarianceThreshold(threshold=0.01)
```

Removes features where almost all values are the same. A feature with variance near 0 carries almost no information — it cannot distinguish one sample from another.

**Example:** If `FTP_COMMAND_RET_CODE` is 0 for 99.9% of rows, its variance ≈ 0. It's useless for separating attack classes.

**Your config:** `threshold: 0.01` — features with variance < 0.01 are dropped first, before any expensive MI or tree computation.

---

## 10. Correlation Filter

```python
threshold: 0.95  # Drop one from each pair with corr > 0.95
```

From Day 1 EDA, you already know 9 feature pairs have correlation ≥ 0.965. After engineering, many new features like `IN_BYTES_log` and `IN_BYTES_sqrt` will also be highly correlated with each other and with `IN_BYTES`.

**The algorithm:**
1. Compute full correlation matrix (144×144)
2. Take upper triangle to avoid duplicate pairs
3. For each column with any correlation > 0.95, drop it
4. Keep the first feature in each correlated pair

**Why drop rather than average:** Dropping is lossless — if two features carry the same information, keeping one loses nothing.

---

## 11. Mutual Information (MI)

```
MI(X, Y) = how much knowing X reduces uncertainty about Y
MI = 0 → X and Y are completely independent
MI > 0 → knowing X helps predict Y
```

Unlike correlation (which only detects linear relationships), MI detects **any** statistical dependence — linear or non-linear.

**In your code:**
```python
mi_scores = mutual_info_classif(X_clean, y_train, discrete_features=False)
```

**Your top result:** `L4_SRC_PORT_x_PROTOCOL` scored 0.0365 MI. This makes intuitive sense — the combination of source port and protocol uniquely identifies attack types (e.g., port 0 + ICMP = ICMP flood, port 53 + UDP = DNS amplification).

---

## 12. Tree-Based Feature Importance

Random Forest trains 100 decision trees and measures how much each feature reduces **Gini impurity** (class mixing) across all splits.

```
importance_i = sum of (impurity_reduction × fraction_of_samples) 
               for all splits on feature_i, averaged across all trees
```

**In your results:** `L4_SRC_PORT_x_L4_DST_PORT` scored highest tree importance (0.0535). Port pairs directly identify service types — SSH (22→any), HTTP (any→80), DNS (any→53) — each maps cleanly to attack or benign.

**Difference from MI:** MI is a statistical measure computed before training. Tree importance is computed by actually training a model — it reflects what the model learned, not just statistics.

---

## 13. Recursive Feature Elimination (RFE)

RFE works backwards — starts with all features, repeatedly removes the least important ones.

**Algorithm:**
1. Train a model on all features
2. Rank features by importance
3. Remove the bottom `step` features (your config: step=5)
4. Repeat until `n_features_to_select` remain
5. Each feature gets a `ranking` — 1 = selected, higher = removed earlier

**In your results:**
- Features with RFE ranking = 1: `L4_SRC_PORT`, `OUT_BYTES`, `LONGEST_FLOW_PKT`, `SRC_TO_DST_AVG_THROUGHPUT`, all interaction features
- Features with RFE ranking = 12 (worst): `PROTOCOL_pow2`, `FTP_COMMAND_RET_CODE`, `Label`, `TCP_FLAGS_percentile`

**Your optimisation:** Pre-filter to top 100 by variance before RFE runs. RFE is O(n_features × n_iterations) — running it on 144 features with step=5 requires ~28 model fits. Pre-filtering to 100 cuts this to ~20 fits.

---

## 14. Ensemble Voting — Combining Selection Methods

Your `combine_selections()` function gives each feature one vote per method that selected it:

```
Feature                      MI   Tree   GB    RFE   Votes
-----------------------------------------------------------------
L4_SRC_PORT_x_PROTOCOL        ✓    ✓      ✓     ✓      4
agg_sum                       ✓    ✓      ✓     ✓      4
OUT_BYTES_binned              ✓    ✓      ✓     ✓      4
PROTOCOL_pow2                 ✗    ✗      ✗     ✗      0  → dropped
```

**Why voting is better than any single method:**
- MI can miss features that are useful in combinations (interaction effects)
- Tree importance is biased toward high-cardinality features
- RFE is expensive and can be unstable on correlated features
- Agreement across methods = stronger signal, lower chance of false selection

**Your min_votes = 2:** Features need at least 2 methods to agree. If fewer than 50 features qualify, the threshold relaxes to fill the quota.

---

## 15. The Leakage Rule — Applied to Feature Engineering

Every transformer in `build_features.py` follows the same rule:

```python
transformer.fit(X_train)              # Learn parameters from train only
X_train = transformer.transform(X_train)
X_test  = transformer.transform(X_test)   # Apply same parameters to test
```

**What gets fitted:**
- `LogTransformer` — which columns to transform (selected by variance on train)
- `BinningTransformer` — bin edges (quantile boundaries calculated on train)
- `InteractionTransformer` — which feature pairs to multiply (top-N by train variance)
- `StandardScaler` — mean and std (calculated on train)

**If you fit on all data:** Test set quantiles, means, and stds influence the transformation → the model indirectly sees test statistics during training → reported accuracy is artificially inflated.

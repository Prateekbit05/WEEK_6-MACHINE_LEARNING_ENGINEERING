# 🧠 TOPICS-TO-LEARN.md — Day 2: Feature Engineering + Feature Selection

All topics tied directly to your code, your dataset, and your actual results.

---

## ✅ Topic 1: Categorical Encoding — OneHot, Label, Target

**Where in your code:** `encode_categorical()` in `build_features.py`

### Key Points

- **OneHot Encoding** creates one binary column per category — used when unique values ≤ 10
- **Label Encoding** maps each category to an integer — used when cardinality is high
- **Target Encoding** replaces category with mean target value — powerful but needs cross-val to avoid leakage
- Your pipeline switches automatically based on `onehot.max_categories` in config
- Unseen test categories are mapped to `'unknown'` before label encoding — prevents `transform()` crash
- LabelEncoder is fit on train only, then applied to test — same leakage rule as scalers

### Important Distinction

| Method | Implies Ordering? | Risk | Best For |
|---|---|---|---|
| OneHot | No | Column explosion | Low-cardinality, linear models |
| Label | Yes (0 < 1 < 2) | Misleads linear models | High-cardinality, tree models |
| Target | No | Leakage if not careful | Any, with CV folds |

**Practice question:**  
Your `IPV4_SRC_ADDR` column has thousands of unique IP addresses. Would you OneHot or Label encode it? What would happen to your feature count with OneHot?

---

## ✅ Topic 2: Numerical Transformations — Log, Sqrt, Power

**Where in your code:** `LogTransformer`, `SqrtTransformer`, `PowerTransformer` in `transformers.py`

### Key Points

- **Log transform:** `log(x + 1)` — compresses right-skewed distributions, handles zeros with +1
- **Sqrt transform:** `sqrt(x)` — gentler compression than log, safe on zero values
- **Power transform:** `x^n` — amplifies differences, opposite effect of log
- All transformers are fit on train data first (select which columns to transform by variance), then applied identically to test

### When Each Applies to Your Data

- `IN_BYTES`, `OUT_BYTES` — extremely right-skewed → `log` or `sqrt`
- `L4_SRC_PORT` — wide range (0–65535), skewed → `log` exposes attack port patterns
- `PROTOCOL` — small integers (1, 6, 17) → `pow2` amplifies differences between TCP/UDP/ICMP

### Formula Reference

```
log_feature  = log(x + 1)
sqrt_feature = sqrt(x)
pow2_feature = x²
```

**Practice question:**  
`IN_BYTES` has values [100, 500, 1000, 50000, 10000000]. Apply log transform manually. What is the range before and after? Why does this help a logistic regression model?

---

## ✅ Topic 3: Interaction and Ratio Features

**Where in your code:** `InteractionTransformer`, `RatioTransformer` in `transformers.py`

### Key Points

- **Interaction features** multiply two features: `A × B` — captures combined effects
- **Ratio features** divide two features: `A / (B + ε)` — captures relative proportions
- Both select top-N features by variance before creating combinations (limits feature explosion)
- Epsilon (small constant) in ratio prevents division by zero

### Why These Matter for Intrusion Detection

- `L4_SRC_PORT_x_PROTOCOL` = port-protocol combination — uniquely fingerprints service types and attack patterns
- `IN_BYTES_div_IN_PKTS` = bytes per packet = average packet size — DDoS uses tiny packets at high rate
- `L4_SRC_PORT_div_IN_BYTES` = port-normalized traffic volume — captures anomalous data volume for specific ports

### The Highest Scoring Feature in Your Dataset

`L4_SRC_PORT_x_PROTOCOL` scored the highest MI overall (0.0365). A single port or single protocol is ambiguous — their combination is not. Port 53 + UDP = DNS, which could be DNS amplification attack. Port 80 + TCP = HTTP, which could be normal or web attack.

**Practice question:**  
You have 6 top features by variance. How many interaction pairs can you create? How many ratio pairs? What is the total?  
*(Hint: n×(n-1)/2 for pairs from n features)*

---

## ✅ Topic 4: Aggregation Features

**Where in your code:** `AggregationTransformer` in `transformers.py`

### Key Points

- Row-level summary statistics computed across all numerical features
- `agg_sum`, `agg_mean`, `agg_std`, `agg_max`, `agg_min`, `agg_median`, `agg_skew`
- These capture the overall "intensity profile" of a network flow in a single number
- `agg_sum` was your top Gradient Boosting feature (importance = 0.210) — by far the most important for GB

### Why `agg_sum` Is So Powerful for Network Data

| Flow Type | agg_sum Profile |
|---|---|
| Benign idle | Low — most features near zero |
| DDoS attack | Extremely high — bytes, packets, flags all maxed |
| Port scan | Medium — many features active at low levels |
| Data exfiltration | High but specific — large bytes, few packets |

**Practice question:**  
A flow has `IN_BYTES=5000`, `OUT_BYTES=200`, `IN_PKTS=10`, `OUT_PKTS=2`, all other features = 0. Calculate `agg_sum`, `agg_mean`, `agg_std`. Is `agg_std` high or low? What does that tell you about the flow?

---

## ✅ Topic 5: Binning Features

**Where in your code:** `BinningTransformer` in `transformers.py`, config `strategy: quantile`

### Key Points

- Converts continuous numerical feature into discrete bins (0, 1, 2, 3, 4)
- **Quantile binning:** Each bin has equal population — 20% of rows per bin with 5 bins
- **Equal-width binning:** Each bin has equal value range — most rows fall in lowest bin for skewed data
- Bin edges are calculated on train set and applied to test (leakage prevention)
- Creates ordinal features that capture "traffic volume tier" rather than exact byte count

### Why `OUT_BYTES_binned` Was Your Top MI Feature (0.0372)

Raw `OUT_BYTES` is extremely skewed — most values are small, a few are enormous. This skewness makes it hard for models to find the right threshold. Quantile binning produces:
- Bin 0: Tiny flows (likely benign idle / scanning)
- Bin 2: Normal traffic
- Bin 4: Extremely large flows (likely DDoS or exfiltration)

Each bin now has 20% of all flows — evenly distributed, easy to learn.

### Quantile vs Equal-Width: The Key Test

```
OUT_BYTES values: [10, 20, 30, 50, 1000000]

Equal-width (5 bins, range 0-1000000):
  Bin 0 (0-200k): [10, 20, 30, 50]   ← 80% of data in one bin
  Bin 4 (800k-1M): [1000000]          ← one outlier dominates

Quantile (5 bins, 20% each):
  Bin 0: [10]
  Bin 1: [20]
  Bin 2: [30]
  Bin 3: [50]
  Bin 4: [1000000]   ← outlier isolated in its own bin
```

**Practice question:**  
Why does quantile binning make your `OUT_BYTES` feature more useful for a logistic regression model than the raw value?

---

## ✅ Topic 6: Feature Selection — Variance Threshold

**Where in your code:** `remove_low_variance()` in `feature_selector.py`

### Key Points

- Features with near-zero variance carry almost no information — same value for almost every row
- Threshold = 0.01 in your config: any feature where variance < 0.01 is dropped immediately
- Computed before any expensive MI or tree methods — saves time
- `FTP_COMMAND_RET_CODE` had near-zero variance (almost all rows = 0 in your dataset) — removed here

### Formula

```
Variance = mean((x - mean(x))²)
If variance < threshold → drop the feature
```

**Practice question:**  
Column `FTP_COMMAND_RET_CODE` is 0 for 99.8% of rows and 200 for 0.2% of rows. Calculate its approximate variance. Will it pass the 0.01 threshold?

---

## ✅ Topic 7: Correlation Filter

**Where in your code:** `remove_correlated()` in `feature_selector.py`

### Key Points

- Correlation threshold = 0.95 in your config
- Uses upper triangle of correlation matrix to avoid duplicate pair checks
- For each pair with corr > 0.95, drops the second feature (keeps the first)
- After engineering, new pairs like `IN_BYTES_log` vs `IN_BYTES_sqrt` will also be highly correlated

### The Algorithm Step by Step

```python
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
```

### Why Upper Triangle Only

The full correlation matrix is symmetric: `corr(A,B) = corr(B,A)`. If you checked both halves, you'd drop both A and B when only dropping one is sufficient. `k=1` in `np.triu` shifts the diagonal up by 1 — excludes self-correlations (always 1.0).

**Practice question:**  
After engineering, `IN_BYTES_log` and `IN_BYTES_sqrt` have correlation = 0.98. Which one gets dropped? What would happen if you kept both in a Logistic Regression model?

---

## ✅ Topic 8: Mutual Information Selection

**Where in your code:** `mutual_information_selection()` in `feature_selector.py`

### Key Points

- MI measures how much knowing a feature reduces uncertainty about the target class
- MI = 0 means feature and target are statistically independent (feature is useless)
- MI > 0 means feature helps predict the target (higher = more useful)
- Detects **non-linear** relationships — unlike correlation which only detects linear
- `discrete_features=False` used since your features are continuous after transformation

### MI vs Pearson Correlation

| Property | Pearson Correlation | Mutual Information |
|---|---|---|
| Detects linear relationships | Yes | Yes |
| Detects non-linear relationships | No | Yes |
| Range | -1 to +1 | 0 to ∞ |
| Computationally expensive | No | Yes |

**Practice question:**  
`PROTOCOL` has MI score = 0.0013 while `L4_SRC_PORT_x_PROTOCOL` scores 0.0365. Why does the interaction feature score 28× higher than the raw feature?

---

## ✅ Topic 9: Recursive Feature Elimination (RFE)

**Where in your code:** `rfe_selection()` in `feature_selector.py`

### Key Points

- RFE works backward: starts with all features, removes the least important in each step
- `step=5` in your code: removes 5 features per iteration
- `n_features_to_select=30`: stops when 30 features remain
- Each feature gets a `ranking`: 1 = kept, higher number = removed earlier
- Pre-filtered to top 100 by variance before RFE runs (computational optimisation)

### RFE Ranking Interpretation from Your Results

- Ranking 1 (best — kept): `L4_SRC_PORT`, `OUT_BYTES`, `agg_sum`, all interaction features
- Ranking 12 (worst — removed first): `PROTOCOL_pow2`, `FTP_COMMAND_RET_CODE`, `Label`, `TCP_FLAGS_percentile`

### Why Some Good Features Still Get Low RFE Ranking

RFE is greedy — it removes based on current importance, not future importance. A feature that's mildly useful alone might become more useful once correlated features are removed. This is why ensemble voting across MI + Tree + GB + RFE is more robust than RFE alone.

**Practice question:**  
Your RFE uses `step=5` and starts with 100 features (after pre-filtering). How many model training rounds does RFE need to reach 30 features? How many rounds would it need with `step=1`?

---

## ✅ Topic 10: Ensemble Voting for Feature Selection

**Where in your code:** `combine_selections()` in `feature_selector.py`

### Key Points

- Each of 4 selection methods (MI, Tree, GB, RFE) votes for its top features
- `min_votes = 2`: feature must appear in at least 2 methods' top lists to be kept
- If fewer than 50 features qualify, threshold relaxes automatically
- Voting is more stable than any single method — reduces selection variance

### Voting Example from Your Results

```
Feature                    MI   Tree  GB    RFE    Votes   Selected?
--------------------------------------------------------------------
L4_SRC_PORT_x_PROTOCOL     ✓    ✓     ✓     ✓      4       YES
agg_sum                    ✓    ✓     ✓     ✓      4       YES
PROTOCOL_pow2              ✗    ✗     ✗     ✗      0       NO
agg_min                    ✗    ✗     ✗     ✗      0       NO
```

### Why Ensemble > Single Method

- MI alone: picks statistically informative features but ignores model-specific learning
- Tree alone: biased toward high-cardinality and continuous features
- RFE alone: expensive, greedy, sensitive to correlated features
- GB alone: single model perspective, may overfit to GB's own biases
- **Voting:** agreement = stronger signal, lower false selection rate

**Practice question:**  
`agg_skew` appears in MI top 50 (score 0.0342) and RFE top 30 (ranking 1) but not in Tree top 50 or GB top 50. How many votes does it have? Does it make the final cut with min_votes=2?

---

## 📌 Quick Revision Checklist

Before Day 3, make sure you can answer:

- [ ] When would you use Label Encoding over OneHot? Name a risk of each.
- [ ] What does `log(x+1)` do to a right-skewed feature? Draw the effect.
- [ ] What is an interaction feature? Give one example from your dataset.
- [ ] Why is `IN_BYTES_div_IN_PKTS` more informative than `IN_BYTES` alone?
- [ ] What does variance threshold remove, and why do it first?
- [ ] What is the difference between MI and Pearson correlation?
- [ ] What does RFE ranking = 1 mean? What does ranking = 12 mean?
- [ ] Why use voting across multiple selection methods?
- [ ] Why fit all transformers on train data only?
- [ ] Your pipeline went 42 → 144 → 50 features. What happened at each step?

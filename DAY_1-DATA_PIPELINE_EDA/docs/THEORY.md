# 📚 THEORY.md — Day 1: Data Pipeline + EDA + Project Architecture

---

## 1. Why ML Projects Need a Pipeline Architecture

A pipeline is a structured, automated sequence of steps that transforms raw data into a model-ready format. Without it, ML projects become brittle — changes in one place break everything else, experiments are hard to reproduce, and bugs are almost impossible to trace.

**The core principle:** Every transformation must be a function, every function must be logged, every output must be versioned.

Your `DataPipeline` class follows this pattern exactly:
- Each step (`clean_data`, `create_train_test_split`, `handle_imbalance`) is isolated
- Every action is logged via `logger`
- Outputs are saved to deterministic paths from `config.yaml`

---

## 2. Train / Validation / Test Split Strategy

### Why Three Splits?

| Split | Used For | Typical Size |
|---|---|---|
| **Train** | Model learns from this | 70–80% |
| **Validation** | Tune hyperparameters | 10–15% |
| **Test** | Final, unbiased evaluation | 10–20% |

Using your test set during tuning causes **data leakage** — your model appears better than it is because it has indirectly seen the test data.

### Your Implementation
Your pipeline uses an 80/20 train-test split with `stratify=True`, which preserves class proportions. This is essential for your dataset since `Benign` is 33% and `Theft` is 0.001% — a random split without stratification could result in no `Theft` samples in the test set at all.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,        # Preserves class %
    random_state=42    # Reproducibility
)
```

### Rule: SMOTE only on training data
You apply `handle_imbalance()` **after** splitting, and only to `X_train`. Applying SMOTE before splitting would let synthetic samples leak into the test set — your model would be evaluated on data it partially generated.

---

## 3. Handling Missing Values

Your pipeline found **0 missing values** in NF-UQ-NIDS-v2, but the code handles three strategies:

### Strategy Comparison

| Strategy | When to Use | Risk |
|---|---|---|
| `median` | Numerical, skewed data | Safe — robust to outliers |
| `mean` | Numerical, normal distribution | Sensitive to outliers |
| `mode` | Categorical columns | Can over-represent one class |
| `zero` | Counts, flags | Only if 0 is meaningful |

### Why median for network data?
Network features like `OUT_BYTES`, `FLOW_DURATION` are heavily right-skewed (a few massive transfers distort the mean). Median is always the safer choice here.

### Drop threshold
Columns missing >50% of values are dropped entirely. Imputing more than half a column introduces more noise than signal.

---

## 4. Outlier Detection

Your EDA found significant outliers — `TCP_WIN_MAX_OUT` had **24.9% IQR outliers**. Two methods are used:

### IQR (Interquartile Range) — Your primary method

```
Lower Bound = Q1 - (1.5 × IQR)
Upper Bound = Q3 + (1.5 × IQR)
where IQR = Q3 - Q1
```

**Strengths:** Distribution-free (no normality assumption), interpretable, robust.  
**Used when:** Data is skewed or you don't know the distribution — perfect for network traffic.

### Z-Score

```
Z = (value - mean) / std_deviation
Outlier if |Z| > 3
```

**Strengths:** Simple, fast.  
**Weakness:** Assumes normal distribution; distorted by the very outliers you're trying to detect.

### Treatment: Capping (Winsorization)
Your pipeline caps outliers at the bounds instead of removing them. This:
- Preserves all rows (no data loss)
- Reduces extreme values' influence on the model
- Keeps the dataset size at 100,000 rows

---

## 5. Data Scaling

Your pipeline supports three scalers configured in `config.yaml`:

### StandardScaler (Z-score normalization)
```
X_scaled = (X - mean) / std
Result: mean=0, std=1
```
Best for: Algorithms sensitive to variance (Logistic Regression, SVM, Neural Networks). Use when data is roughly normally distributed.

### MinMaxScaler
```
X_scaled = (X - X_min) / (X_max - X_min)
Result: values between 0 and 1
```
Best for: Neural networks, image data. Sensitive to outliers — a single extreme value compresses everything else.

### RobustScaler
```
X_scaled = (X - median) / IQR
```
Best for: Data with many outliers (like your network traffic data). Uses median and IQR instead of mean and std.

**For NF-UQ-NIDS-v2:** RobustScaler would actually be the best choice given the 24.9% outlier rate in some columns.

---

## 6. Class Imbalance — SMOTE

Your dataset has a **32,986:1 imbalance ratio** (Benign vs. Theft). This is extreme. Without handling it, a model that predicts "Benign" for everything achieves 33% accuracy while detecting zero attacks.

### SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE creates **synthetic samples** for minority classes by:
1. Taking a minority sample
2. Finding its k nearest neighbours (default k=5)
3. Creating a new point on the line between the sample and a random neighbour

```
New_Sample = Sample + random(0,1) × (Neighbour - Sample)
```

### Why not just duplicate minority samples?
Simple duplication (Random Oversampling) makes the model memorize those exact points. SMOTE creates variation, helping the model learn the general region of minority classes.

### Your SMOTE implementation
```python
# k_neighbors adjusted for very small classes
k_neighbors = min(5, min_class_size - 1)
sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
```

For classes like `Shellcode` (2 samples) or `Theft` (1 sample), SMOTE falls back to RandomOverSampler automatically.

---

## 7. Dataset Versioning

### Why version data?
In ML, a model is only reproducible if you can recreate the exact dataset it trained on. Dataset versioning answers: *"Which version of the data produced this result?"*

### Your approach: DataFrame hashing
```python
data_hash = calculate_dataframe_hash(df)
# Saves: version_a3f92c1b.json with rows, columns, timestamp
```

A hash changes if even one value in the dataset changes — giving you exact reproducibility.

### Alternative: DVC (Data Version Control)
DVC works like Git for data files — tracks large files in remote storage (S3, GCS) and stores only metadata in Git. For production systems with 10+ GB datasets, DVC is the standard tool.

---

## 8. Chunked Processing

Your pipeline reads data in 50,000-row chunks instead of loading everything at once:

```python
for chunk in pd.read_csv(filepath, chunksize=50000):
    # Process chunk
    # Garbage collect every 10 chunks
```

**Why it matters:** The NF-UQ-NIDS-v2 full dataset is several GB. Loading it entirely into RAM would crash most machines. Chunking keeps memory usage constant regardless of file size.

---

## 9. EDA Key Findings for NF-UQ-NIDS-v2

### Correlation Alert
9 feature pairs have correlation ≥ 0.965:
- `OUT_BYTES` and `NUM_PKTS_1024_TO_1514_BYTES` = **1.000** (perfectly correlated)
- `LONGEST_FLOW_PKT` and `MAX_IP_PKT_LEN` = **1.000**
- `ICMP_TYPE` and `ICMP_IPV4_TYPE` = **1.000**

Perfectly correlated features carry identical information. Keeping both adds noise, increases training time, and can destabilize some models (especially linear ones).

**Action for Day 2:** Drop one feature from each perfectly correlated pair using a correlation threshold filter.

### Class Imbalance
With `Shellcode` having 2 samples and `Theft` having 1 sample, these classes cannot be modelled reliably. Your pipeline correctly removes classes below `min_samples_per_class=10` before SMOTE.

---

## 10. The Config-Driven Design Pattern

Your `DataPipeline` reads all parameters from `config.yaml` instead of hardcoding them. This is a production best practice:

- **Reproducibility:** Config file can be committed to Git alongside results
- **Flexibility:** Change behaviour without touching code
- **Auditability:** Every experiment's config is traceable

The `DataConfig` and `PreprocessingConfig` dataclasses enforce type safety — if you pass a string where a float is expected, Python raises an error immediately rather than failing silently during processing.
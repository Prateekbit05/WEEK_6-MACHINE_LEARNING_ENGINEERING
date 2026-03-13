# 🧠 TOPICS-TO-LEARN.md — Day 1: Data Pipeline + EDA

A self-study guide tied directly to what your pipeline does. Each topic links to where it appears in your code.

---

## ✅ Topic 1: Train / Validation / Test Splitting

**Where in your code:** `create_train_test_split()` in `data_pipeline.py`

**What to understand:**
- Why test data must never be seen during training or tuning
- What stratification does and when it matters
- The difference between validation set and test set
- Why `random_state=42` makes results reproducible

**Practice question:**  
Your dataset has 100,000 rows with a 20% test split. After SMOTE, your training set grows to 160,000 rows. What is your actual test set size? Why doesn't SMOTE affect it?

---

## ✅ Topic 2: Handling Missing Values

**Where in your code:** `_handle_missing_values()` in `data_pipeline.py`

**What to understand:**
- Difference between mean, median, and mode imputation
- What `drop_threshold=0.5` means and why 50% is a reasonable cutoff
- Why `SimpleImputer` is preferred over manual `fillna()` in pipelines
- The concept of fitting imputers on train data only (prevent leakage)

**Practice question:**  
Column `FLOW_DURATION` has 200 extreme values of 999999. You impute with mean. What happens to the imputed value for other missing rows? Would median have been better?

---

## ✅ Topic 3: Outlier Detection — IQR vs Z-Score

**Where in your code:** `_handle_outliers()`, `detect_outliers_iqr()`, `detect_outliers_zscore()` in `helpers.py`

**What to understand:**
- How to calculate Q1, Q3, IQR manually
- What the 1.5× multiplier means (and when to use 3×)
- Why Z-score assumes normality (and why that's a problem for network data)
- Difference between removing outliers vs capping (winsorization)

**Formula to memorize:**
```
IQR = Q3 - Q1
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
```

**Practice question:**  
`TCP_WIN_MAX_OUT` has 24.9% IQR outliers but only 6.34% Z-score outliers. What does this tell you about the distribution of this feature?

---

## ✅ Topic 4: Data Scaling

**Where in your code:** `PreprocessingConfig.scaling_method`, called before model training

**What to understand:**
- Why tree-based models (Random Forest, XGBoost) don't need scaling
- Why Logistic Regression and Neural Networks need scaling
- The math behind StandardScaler, MinMaxScaler, RobustScaler
- Why RobustScaler is the best choice for your dataset

**Scaling methods summary:**
| Scaler | Formula | Best For |
|---|---|---|
| StandardScaler | (x - mean) / std | Normal distributions |
| MinMaxScaler | (x - min) / (max - min) | Neural nets, no outliers |
| RobustScaler | (x - median) / IQR | Data with outliers |

**Practice question:**  
Your dataset has a column `DURATION` with values [1, 2, 3, 4, 9999]. Apply MinMaxScaler mentally. What happens to values 1–4 after scaling?

---

## ✅ Topic 5: Class Imbalance — SMOTE

**Where in your code:** `handle_imbalance()` in `data_pipeline.py`

**What to understand:**
- Why accuracy is a misleading metric for imbalanced data
- What SMOTE does step by step (k-NN based synthetic generation)
- Why SMOTE is applied only to training data
- When to use SMOTE vs class weights vs undersampling

**SMOTE step by step:**
1. Pick a minority sample point A
2. Find its k nearest neighbours in feature space
3. Pick one neighbour B randomly
4. Create new point: `A + rand(0,1) × (B - A)`

**Practice question:**  
Your `Shellcode` class has only 2 samples. With k=5 SMOTE, what happens? What does your pipeline do instead, and why?

---

## ✅ Topic 6: Dataset Versioning

**Where in your code:** `version_data()` in `data_pipeline.py`

**What to understand:**
- Why ML experiments need data versioning (not just code versioning)
- How file hashing works to track dataset changes
- What DVC (Data Version Control) does and how it differs from folder hashing
- What gets stored in a `version_*.json` file

**Practice question:**  
You run the pipeline twice on the same raw file. Are the version hashes the same? What if you change one row in the raw file?

---

## ✅ Topic 7: Chunked / Memory-Efficient Data Loading

**Where in your code:** `create_sample()` — `pd.read_csv(filepath, chunksize=50000)`

**What to understand:**
- Why you can't always load a whole CSV into RAM
- How `chunksize` in pandas works
- What `force_garbage_collection()` does between chunks
- How to estimate total rows from file size + sample

**Practice question:**  
A 5 GB CSV with 50 columns is being read in 50,000-row chunks. If each chunk takes 3 seconds to process, estimate total processing time. How would you reduce this?

---

## ✅ Topic 8: EDA — Correlation Analysis

**Where in your EDA:** `Highly Correlated Features` section, `correlation_matrix_full.png`

**What to understand:**
- What Pearson correlation measures (linear relationship, -1 to +1)
- Why correlation of 1.0 means two features are perfectly redundant
- How correlation threshold filtering works in feature selection
- Difference between correlation and causation

**Your finding:**  
`OUT_BYTES` and `NUM_PKTS_1024_TO_1514_BYTES` have correlation = 1.000. This means knowing one tells you nothing new when you also have the other.

**Practice question:**  
You have 42 numerical features. After applying a 0.95 correlation threshold filter, 9 pairs are flagged. How many features would you drop, and which one in each pair do you keep?

---

## ✅ Topic 9: Config-Driven Pipeline Design

**Where in your code:** `DataConfig`, `PreprocessingConfig` dataclasses + `config.yaml`

**What to understand:**
- Why hardcoding parameters in code is an anti-pattern
- What a YAML config file is and how `yaml.safe_load()` reads it
- Why Python `dataclass` is better than plain dictionaries for configs
- How to make experiments reproducible through config versioning

---

## ✅ Topic 10: Data Quality Checks

**Where in your code:** `_remove_duplicates()`, `_drop_high_missing_columns()`, `validate_dataframe()`

**What to understand:**
- The difference between structural errors (wrong dtypes) and content errors (wrong values)
- Why duplicate rows are dangerous for model training (inflated metrics)
- What `optimize_dtypes()` does to reduce memory (e.g., int64 → int8 where possible)

**Your finding:**  
0 missing values and 0 duplicates in NF-UQ-NIDS-v2. This is unusual — real datasets almost always have both. Your pipeline handles them anyway, which is the right approach.

---

## 📌 Quick Revision Checklist

Before Day 2, make sure you can answer:

- [ ] Why is test data split done before SMOTE?
- [ ] What is the IQR formula for outlier bounds?
- [ ] Name 3 imputation strategies and when to use each
- [ ] What does stratify=True do in train_test_split?
- [ ] What is the imbalance ratio in your dataset and why is it a problem?
- [ ] What does a data version hash tell you?
- [ ] Why is median better than mean for your network traffic features?
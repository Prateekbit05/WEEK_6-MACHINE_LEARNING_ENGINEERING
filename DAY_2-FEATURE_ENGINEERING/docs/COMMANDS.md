# ⚡ COMMANDS.md — Day 2: Feature Engineering + Feature Selection

All commands to run, verify, and debug the Day 2 feature pipeline.

---

## 🔧 Setup

```bash
# Activate virtual environment
source venv/bin/activate           # Linux/Mac
venv\Scripts\activate              # Windows

# Install Day 2 dependencies
pip install scikit-learn pandas numpy matplotlib seaborn pyyaml

# Verify imports
python -c "from sklearn.feature_selection import mutual_info_classif, RFE; print('OK')"
```

---

## 🚀 Run the Full Day 2 Pipeline

```bash
# From project root
cd ~/Documents/.../DAY_2-FEATURE_ENGINEERING

# Step 1 — Feature Engineering (42 → 144 features)
python src/features/build_features.py

# Step 2 — Feature Selection (144 → 50 features)
python src/features/feature_selector.py

# Run both steps back to back
python src/features/build_features.py && python src/features/feature_selector.py
```

---

## 🔍 Run Individual Steps

```bash
# Test FeatureEngineer initialisation only
python -c "
from src.features.build_features import FeatureEngineer
fe = FeatureEngineer('src/config/config.yaml')
print('Config loaded OK')
"

# Load data only (no transforms)
python -c "
from src.features.build_features import FeatureEngineer
fe = FeatureEngineer('src/config/config.yaml')
result = fe.load_data()
if result:
    train, test = result
    print('Train:', train.shape, '| Test:', test.shape)
"

# Run engineering only, skip selection
python -c "
from src.features.build_features import FeatureEngineer
fe = FeatureEngineer('src/config/config.yaml')
result = fe.run_pipeline()
if result:
    X_train, X_test, y_train, y_test = result
    print('Features after engineering:', X_train.shape[1])
"

# Run selection only (requires build_features to have run first)
python -c "
from src.features.feature_selector import FeatureSelector
fs = FeatureSelector('src/config/config.yaml')
result = fs.run_selection_pipeline()
if result:
    X_train, X_test, y_train, y_test = result
    print('Final features:', X_train.shape[1])
"
```

---

## 🔎 Verify Outputs

```bash
# Check engineered features exist
ls -lh src/data/processed/X_train_engineered.csv
ls -lh src/data/processed/X_test_engineered.csv

# Check final selected features exist
ls -lh src/data/processed/final_features/

# Inspect feature count at each stage
python -c "
import pandas as pd
eng  = pd.read_csv('src/data/processed/X_train_engineered.csv')
fin  = pd.read_csv('src/data/processed/final_features/X_train_final.csv')
print('Engineered features:', eng.shape[1])
print('Final features:     ', fin.shape[1])
"

# View final 50 selected features
python -c "
import json
with open('src/features/feature_list.json') as f:
    fl = json.load(f)
print('N features:', fl['n_features'])
for feat in fl['features']:
    print(' -', feat)
"

# Check top 10 by mutual information
python -c "
import json
with open('src/features/feature_list.json') as f:
    fl = json.load(f)
mi = fl['feature_scores']['mutual_information']
top10 = sorted(mi.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, score in top10:
    print(f'{feat:40s} {score:.4f}')
"

# Verify no NaN in final features
python -c "
import pandas as pd
df = pd.read_csv('src/data/processed/final_features/X_train_final.csv')
print('NaN count:', df.isnull().sum().sum())
print('Shape:', df.shape)
"
```

---

## 📊 Check Feature Importance Plots

```bash
# List all generated plots
ls src/notebooks/plots/

# Open all importance plots (Linux)
eog src/notebooks/plots/importance_*.png

# Open correlation heatmap
eog src/notebooks/plots/feature_correlation.png

# Open voting results
eog src/notebooks/plots/feature_selection_votes.png
```

---

## 🧪 Test Individual Transformers

```bash
# Test log transformer
python -c "
import pandas as pd
import numpy as np
from src.features.transformers import LogTransformer

df = pd.DataFrame({'A': [1,2,3,100,1000], 'B': [0,5,10,50,500]})
lt = LogTransformer(n_features=2)
lt.fit(df)
result = lt.transform(df)
print('Log features added:', [c for c in result.columns if '_log' in c])
"

# Test binning transformer
python -c "
import pandas as pd
from src.features.transformers import BinningTransformer

df = pd.DataFrame({'A': range(100), 'B': range(100,200)})
bt = BinningTransformer(n_features=2, n_bins=5, strategy='quantile')
bt.fit(df)
result = bt.transform(df)
print('Binned features:', [c for c in result.columns if '_binned' in c])
"
```

---

## 🔎 Compare Feature Importance Across Methods

```bash
python -c "
import json
with open('src/features/feature_list.json') as f:
    fl = json.load(f)

methods = ['mutual_information', 'tree_based', 'gradient_boosting']
print(f'{'Feature':<40} {'MI':>8} {'Tree':>8} {'GB':>8}')
print('-' * 68)

# Show selected features only
for feat in fl['features'][:15]:
    scores = []
    for m in methods:
        s = fl['feature_scores'][m].get(feat, 0)
        scores.append(f'{s:.4f}')
    print(f'{feat:<40} {scores[0]:>8} {scores[1]:>8} {scores[2]:>8}')
"
```

---

## 🐛 Troubleshooting

```bash
# If build_features fails with import error for transformers
python -c "from src.features.transformers import LogTransformer; print('OK')"

# If FeatureEngineer says train data not found
ls src/data/inputs/
# Make sure Day 1 train.csv is copied to inputs/ folder as configured in config.yaml

# If mutual_info_classif crashes with negative values
# It auto-shifts to non-negative internally — check your config for:
# feature_selection.mutual_info.shift_non_negative: true

# If RFE is too slow (timeout on 144 features)
# Reduce in config.yaml:
# feature_selection.rfe.n_features_to_select: 20   (default: 30)
# The pipeline pre-filters to top 100 by variance before RFE runs

# If plots directory does not exist
mkdir -p src/notebooks/plots

# Check logs for detailed errors
tail -100 src/logs/*.log
```

---

## 📦 Output File Summary

```bash
# All Day 2 outputs
src/data/processed/X_train_engineered.csv       # 144 features, training
src/data/processed/X_test_engineered.csv        # 144 features, test
src/data/processed/final_features/
    X_train_final.csv                           # 50 features, training
    X_test_final.csv                            # 50 features, test
src/features/
    feature_list.json                           # 50 features + all scores
    engineered_features.json                    # Engineering metadata
src/notebooks/plots/
    importance_mutual_information.png
    importance_tree_based.png
    importance_gradient_boosting.png
    feature_correlation.png
    feature_selection_votes.png
```

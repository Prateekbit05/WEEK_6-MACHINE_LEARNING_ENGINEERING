# ⌨️ COMMANDS.md — Day 3 Quick Reference

All commands to run, debug, and inspect the Day 3 training pipeline.

---

## 🚀 Run Pipeline

```bash
# Full training pipeline (recommended)
python src/training/train.py

# From project root (explicit)
cd /path/to/day-3
python src/training/train.py
```

---

## 📦 Install Dependencies

```bash
# Install all at once
pip install scikit-learn xgboost lightgbm pandas numpy matplotlib seaborn pyyaml

# Or from requirements file (if present)
pip install -r requirements.txt

# Verify installs
python -c "import sklearn, xgboost, lightgbm; print('All OK')"
```

---

## 🔍 Inspect Outputs

```bash
# View saved metrics
cat src/evaluation/metrics.json

# Pretty-print metrics
python -c "
import json
with open('src/evaluation/metrics.json') as f:
    data = json.load(f)
print('Best Model:', data['best_model'])
print('Test Metrics:')
for k, v in data['test_metrics'].items():
    print(f'  {k}: {v:.4f}')
"

# Check model file exists and size
ls -lh src/models/best_model.pkl

# View model comparison report
cat reports/MODEL-COMPARISON.md

# List all generated plots
ls -lh plots/
```

---

## 🧪 Load & Test Saved Model

```bash
python -c "
import pickle
import pandas as pd

# Load model
with open('src/models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('Model loaded:', type(model))
print('Steps:', model.steps)

# Quick prediction test
df = pd.read_csv('inputs/X_test_final.csv')
target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

preds = model.predict(X[:5])
print('Sample predictions:', preds)
print('Actual labels:     ', y[:5].values)
"
```

---

## 📊 Re-plot Confusion Matrix

```bash
python -c "
import pickle, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

with open('src/models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('inputs/X_test_final.csv')
X = df.drop(columns=['target'])
y = df['target']

preds = model.predict(X)
cm = confusion_matrix(y, preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_norm, annot=False, cmap='Blues')
plt.title('Confusion Matrix (Normalized)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/cm_replot.png', dpi=150)
print('Saved: plots/cm_replot.png')
"
```

---

## 📈 Print Full CV Results Table

```bash
python -c "
import json

with open('src/evaluation/metrics.json') as f:
    data = json.load(f)

print(f\"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'AUC':>6} {'Time':>7}\")
print('-' * 65)
for name, res in data['cv_results'].items():
    print(f\"{name:<22} {res['accuracy_mean']:>6.4f} {res['precision_mean']:>6.4f} {res['recall_mean']:>6.4f} {res['f1_mean']:>6.4f} {res['roc_auc_mean']:>6.4f} {res['training_time']:>6.1f}s\")
print()
print(f\"Best model: {data['best_model']}\")
"
```

---

## 🛠️ Debug & Troubleshoot

```bash
# Check data shape and class distribution
python -c "
import pandas as pd
df = pd.read_csv('inputs/X_train_final.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist()[:5], '...')
print('Target distribution:')
print(df['target'].value_counts().sort_index())
"

# Check for NaN values
python -c "
import pandas as pd
df = pd.read_csv('inputs/X_train_final.csv')
print('NaN counts:')
print(df.isnull().sum().sum(), 'total NaNs')
"

# Verify label range (must start at 0)
python -c "
import pandas as pd
df = pd.read_csv('inputs/X_train_final.csv')
print('Label min:', df['target'].min())
print('Label max:', df['target'].max())
print('Unique classes:', df['target'].nunique())
"

# Run with verbose logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from src.training.train import ModelTrainer
t = ModelTrainer()
t.run_pipeline()
"
```

---

## 🗂️ File Locations Summary

| What | Where |
|------|-------|
| Run pipeline | `python src/training/train.py` |
| Best model | `src/models/best_model.pkl` |
| All metrics | `src/evaluation/metrics.json` |
| Comparison report | `reports/MODEL-COMPARISON.md` |
| Confusion matrix | `plots/cm_lightgbm.png` |
| Model bar chart | `plots/model_comparison.png` |

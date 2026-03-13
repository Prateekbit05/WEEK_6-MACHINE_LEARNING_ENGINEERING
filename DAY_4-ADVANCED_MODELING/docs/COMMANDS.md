# ⌨️ COMMANDS.md — Day 4 Quick Reference

All commands to run, debug, and inspect the Day 4 tuning and explainability pipeline.

---

## 🚀 Run the Full Pipeline

```bash
# Step 1 — Tune with Bayesian optimization (Optuna, recommended)
python src/training/tuning.py

# Step 2 — Run SHAP analysis on best saved model
python src/evaluation/shap_analysis.py
```

---

## 📦 Install Dependencies

```bash
# All Day 4 dependencies
pip install optuna shap xgboost lightgbm scikit-learn joblib pandas numpy matplotlib pyyaml

# Verify installs
python -c "import optuna, shap, xgboost; print('All OK')"

# Check Optuna version
python -c "import optuna; print(optuna.__version__)"

# Check SHAP version
python -c "import shap; print(shap.__version__)"
```

---

## 🎛️ Run Tuning with Different Methods

```bash
# Bayesian (Optuna) — best quality, smart search
python -c "
from src.training.tuning import HyperparameterTuner
t = HyperparameterTuner()
t.run_pipeline(method='bayesian')
"

# Random Search — fast, good for large spaces
python -c "
from src.training.tuning import HyperparameterTuner
t = HyperparameterTuner()
t.run_pipeline(method='random_search')
"

# Grid Search — exhaustive, slow
python -c "
from src.training.tuning import HyperparameterTuner
t = HyperparameterTuner()
t.run_pipeline(method='grid_search')
"
```

---

## 🔍 Inspect Tuning Results

```bash
# View raw results file
cat outputs/tuning_results.json

# Pretty-print with Python
python -c "
import json
with open('outputs/tuning_results.json') as f:
    data = json.load(f)
for model, res in data.items():
    print(f'{model}:')
    print(f'  CV Score:      {res[\"best_cv_score\"]:.4f}')
    print(f'  Test Accuracy: {res[\"test_accuracy\"]:.4f}')
    print(f'  Best Params:   {res[\"best_params\"]}')
    print()
"

# View final tuning comparison (baseline vs tuned)
cat tuning/results.json

# Compare baseline vs tuned F1
python -c "
import json
with open('tuning/results.json') as f:
    r = json.load(f)
print(f'Baseline F1:  {r[\"baseline\"][\"f1\"]:.4f}')
print(f'Tuned F1:     {r[\"tuned\"][\"f1\"]:.4f}')
print(f'Improvement:  {r[\"improvement_f1\"]:+.4f}')
print(f'Best Params:  {r[\"best_params\"]}')
"
```

---

## 🤖 Load and Use Saved Models

```bash
# Load and predict with tuned Random Forest
python -c "
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

model = joblib.load('outputs/models/best_random_forest.joblib')
X_test = pd.read_csv('inputs/X_test_final.csv')
y_test = pd.read_csv('inputs/y_test.csv').iloc[:, 0]

preds = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, preds):.4f}')
print(f'F1 Score: {f1_score(y_test, preds, average=\"macro\"):.4f}')
"

# Check all saved models
ls -lh outputs/models/

# Inspect model parameters
python -c "
import joblib
model = joblib.load('outputs/models/best_random_forest.joblib')
print(model.get_params())
"
```

---

## 📊 SHAP Analysis Commands

```bash
# Run SHAP analysis (generates plots + CSV)
python src/evaluation/shap_analysis.py

# View top features from saved CSV
python -c "
import pandas as pd
df = pd.read_csv('outputs/shap_feature_importance.csv')
print(df.head(10).to_string(index=False))
"

# List all generated plots
ls -lh outputs/plots/

# Re-run SHAP on more samples (edit max_samples)
python -c "
from src.evaluation.shap_analysis import SHAPAnalyzer
analyzer = SHAPAnalyzer()
analyzer.load_model_and_data()
analyzer.compute_shap_values(max_samples=500)  # more samples
analyzer.plot_summary()
analyzer.plot_feature_importance()
analyzer.save_feature_importance()
"
```

---

## 🧪 Run Optuna with Custom Trials

```bash
# Run Optuna with more trials (better optimisation)
python -c "
from src.training.tuning import HyperparameterTuner
t = HyperparameterTuner()
t.load_data()
t.tune_with_optuna(n_trials=100)  # default is 30
t.save_results()
"

# Visualise Optuna study (if optuna[visualization] installed)
pip install optuna[visualization] plotly
python -c "
import optuna
# Load from storage if saved, or re-run and capture study:
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)
print('Install plotly and uncomment above for visual dashboard')
"
```

---

## 🛠️ Debug & Troubleshoot

```bash
# Check data files exist
ls -lh inputs/

# Verify y_train column name
python -c "
import pandas as pd
df = pd.read_csv('inputs/y_train.csv')
print('Columns:', df.columns.tolist())
print('Sample:', df.head(3))
"

# Test SHAP is working
python -c "
import shap
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier

X, y = sklearn.datasets.load_iris(return_X_y=True)
m = RandomForestClassifier(n_estimators=10).fit(X, y)
e = shap.TreeExplainer(m)
sv = e.shap_values(X[:5])
print('SHAP values shape:', [s.shape for s in sv])
print('SHAP working correctly!')
"

# Check for models before running SHAP
python -c "
from pathlib import Path
models = list(Path('outputs/models').glob('*.joblib'))
if models:
    print('Found models:', [m.name for m in models])
else:
    print('No models found — run tuning.py first!')
"
```

---

## 🗂️ File Locations Summary

| What | Where |
|------|-------|
| Run tuning | `python src/training/tuning.py` |
| Run SHAP | `python src/evaluation/shap_analysis.py` |
| Tuned RF model | `outputs/models/best_random_forest.joblib` |
| Tuned XGB model | `outputs/models/best_xgboost.joblib` |
| Tuning metrics | `outputs/tuning_results.json` |
| Final comparison | `tuning/results.json` |
| SHAP summary plot | `outputs/plots/shap_summary.png` |
| SHAP importance plot | `outputs/plots/shap_importance.png` |
| Feature importance CSV | `outputs/shap_feature_importance.csv` |

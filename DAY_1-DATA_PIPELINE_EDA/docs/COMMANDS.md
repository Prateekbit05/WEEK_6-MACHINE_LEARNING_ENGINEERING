# ⚡ COMMANDS.md — Day 1: Data Pipeline

All commands to set up, run, and verify the Day 1 data pipeline.

---

## 🔧 Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import pandas, sklearn, imblearn; print('All good!')"
```

---

## 📂 Project Initialization

```bash
# Create the full folder structure
mkdir -p src/data/raw \
         src/data/processed \
         src/data/external \
         src/data/metadata \
         src/notebooks \
         src/features \
         src/pipelines \
         src/models \
         src/training \
         src/evaluation \
         src/deployment \
         src/monitoring \
         src/utils \
         src/config \
         src/logs
```

---

## 🚀 Running the Pipeline

```bash
# Run the complete Day 1 data pipeline
python src/pipelines/data_pipeline.py

# Run with a custom config file
python -c "
from src.pipelines.data_pipeline import DataPipeline
pipeline = DataPipeline('src/config/config.yaml')
results = pipeline.run_pipeline()
print(results)
"
```

---

## 🔍 Individual Pipeline Steps

```bash
# Step 1 — Check dataset info without loading it all
python -c "
from src.pipelines.data_pipeline import DataPipeline
p = DataPipeline('src/config/config.yaml')
info = p.get_dataset_info()
print(info)
"

# Step 2 — Create 100k sample
python -c "
from src.pipelines.data_pipeline import DataPipeline
p = DataPipeline('src/config/config.yaml')
df = p.create_sample()
print(df.shape)
"

# Step 3 — Clean data only
python -c "
import pandas as pd
from src.pipelines.data_pipeline import DataPipeline
p = DataPipeline('src/config/config.yaml')
df = pd.read_csv('src/data/processed/sample.csv')
clean = p.clean_data(df)
print(clean.shape)
"

# Step 4 — Create train/test split
python -c "
from src.pipelines.data_pipeline import DataPipeline
p = DataPipeline('src/config/config.yaml')
X_train, X_test, y_train, y_test = p.create_train_test_split()
print(X_train.shape, X_test.shape)
"
```

---

## 📓 Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Or run EDA notebook directly as a script
jupyter nbconvert --to notebook --execute src/notebooks/EDA.ipynb

# Export EDA notebook to HTML report
jupyter nbconvert --to html src/notebooks/EDA.ipynb --output src/reports/EDA_report.html
```

---

## 🔎 Data Verification

```bash
# Check processed files exist
ls -lh src/data/processed/

# Quick data preview
python -c "
import pandas as pd
df = pd.read_csv('src/data/processed/final.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df.head(3))
"

# Check class distribution
python -c "
import pandas as pd
df = pd.read_csv('src/data/processed/final.csv')
print(df['Attack'].value_counts())
"

# Check train/test shapes
python -c "
import pandas as pd
train = pd.read_csv('src/data/processed/train.csv')
test  = pd.read_csv('src/data/processed/test.csv')
print('Train:', train.shape, '| Test:', test.shape)
"
```

---

## 🧹 Utilities

```bash
# Check for missing values
python -c "
import pandas as pd
df = pd.read_csv('src/data/processed/final.csv')
print(df.isnull().sum().sum(), 'missing values')
"

# Check for duplicates
python -c "
import pandas as pd
df = pd.read_csv('src/data/processed/final.csv')
print(df.duplicated().sum(), 'duplicates')
"

# View pipeline results log
cat src/data/processed/pipeline_results.json

# View data version hash
ls src/data/metadata/
cat src/data/metadata/version_*.json
```

---

## 🐛 Troubleshooting

```bash
# If imblearn (SMOTE) is not installed
pip install imbalanced-learn

# If memory error on large file — reduce chunk_size in config.yaml
# chunk_size: 10000   (default: 50000)

# If config not found
ls src/config/config.yaml

# Check logs for errors
cat src/logs/*.log
tail -50 src/logs/pipeline.log
```
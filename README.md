# IEEE Fraud Detection Project

This project engineering version organizes the IEEE-CIS Fraud Detection
research workflow into reusable modules, scripts, notebooks, and experiment outputs.

## Workflow

1. Environment setup
2. EDA
3. Data processing
4. Feature engineering
5. WOE-IV analysis and feature selection
6. Model training: WOE-LR, Random Forest, XGBoost
7. Evaluation: AUC, PR-AUC, KS, Precision, Recall, F1, TopK Lift
8. Explainability: TreeSHAP, importance, summary, mechanism-level aggregation

## Setup

Core runtime environment:

```powershell
cd "D:\Vs code\ieee-fraud-project"
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

For tests and development:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

For notebooks:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-notebook.txt
```

## Data

Place Kaggle IEEE-CIS files under:

`data/raw/ieee-fraud-detection/`

Required training files:

- `train_transaction.csv`
- `train_identity.csv`

Optional official test files:

- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

## Commands

Prepare time-based train/valid/test parquet files:

```powershell
.\.venv\Scripts\python scripts/prepare_data.py --config configs/base.yaml
```

Run the full four-scheme experiment with all enabled models:

```powershell
.\.venv\Scripts\python scripts/train.py --config configs/base.yaml
```

Training progress is written to the console and to:

`outputs/logs/training.log`

Run only XGBoost schemes:

```powershell
.\.venv\Scripts\python scripts/train.py --config configs/xgb.yaml
```

Generate SHAP outputs for the main scheme:

```powershell
.\.venv\Scripts\python scripts/explain.py --config configs/base.yaml --experiment iv_002_050
```

## Output layout

- `outputs/tables/iv_summary.csv`
- `outputs/tables/risk_feature_system.csv`
- `outputs/tables/all_metrics_train_valid_test.csv`
- `outputs/tables/all_topk_train_valid_test_with_lift.csv`
- `outputs/<scheme>/models/*.joblib`
- `outputs/<scheme>/metrics/*.csv`
- `outputs/<scheme>/predictions/*.csv`
- `outputs/<scheme>/shap/*.png`

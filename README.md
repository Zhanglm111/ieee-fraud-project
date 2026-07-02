# IEEE Fraud Detection Project

This project turns the IEEE-CIS Fraud Detection experiment into a reusable
machine-learning engineering project.

## Workflow

1. Project initialization
2. Environment setup
3. Exploratory data analysis
4. Data processing
5. Feature engineering
6. IV analysis and feature selection
7. Model training
8. Model evaluation
9. Model explainability with SHAP
10. Experiment management

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

Put Kaggle CSV files under `data/raw/ieee-fraud-detection/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

## Main commands

```powershell
python scripts/prepare_data.py --config configs/xgb.yaml
python scripts/train.py --config configs/xgb.yaml
python scripts/evaluate.py --config configs/xgb.yaml --experiment iv_002_050
python scripts/explain.py --config configs/xgb.yaml --experiment iv_002_050
```

The default configuration reproduces the paper-oriented experiment:

- IV >= 0.02
- 0.02 <= IV <= 0.50
- 0.10 <= IV <= 0.50
- all candidate features

Outputs are organized by experiment under `outputs/<experiment_name>/`.

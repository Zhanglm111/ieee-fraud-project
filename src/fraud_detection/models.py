from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .utils import ensure_dir


MODEL_LABELS = {"woe_lr": "WOE-LR", "lr": "LR", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def make_model(name: str, params: dict | None = None, random_state: int = 42):
    params = dict(params or {})
    if name in {"lr", "woe_lr"}:
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)
    if name == "random_forest":
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)
    if name == "xgboost":
        params.setdefault("random_state", random_state)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model name: {name}")


def fit_xgboost(model, X_train, y_train, X_valid, y_valid, early_stopping_rounds: int | None = None):
    if early_stopping_rounds:
        try:
            return model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        except TypeError:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
    return model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)

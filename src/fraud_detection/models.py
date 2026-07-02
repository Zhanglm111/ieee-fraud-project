from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .utils import ensure_dir


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


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)

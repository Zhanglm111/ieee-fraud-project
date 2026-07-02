from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .utils import ensure_dir


def sample_for_shap(X: pd.DataFrame, sample_size: int, random_state: int, cache_path: str | Path | None = None):
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)["RowIndex"].tolist()
            if pd.api.types.is_integer_dtype(X.index):
                cached = [int(x) for x in cached]
            available = [idx for idx in cached if idx in X.index]
            if len(available) >= min(sample_size, len(X)):
                return X.loc[available[: min(sample_size, len(X))]]
    sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    if cache_path is not None:
        ensure_dir(cache_path.parent)
        pd.DataFrame({"RowIndex": sample.index}).to_csv(cache_path, index=False, encoding="utf-8-sig")
    return sample


def compute_tree_shap(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    if hasattr(values, "values"):
        values = values.values
    if len(values.shape) == 3:
        values = values[:, :, 1]
    return np.asarray(values)


def shap_importance_table(
    X: pd.DataFrame,
    values: np.ndarray,
    feature_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Feature": X.columns,
            "MeanSHAP": values.mean(axis=0),
            "MeanAbsSHAP": np.abs(values).mean(axis=0),
            "MedianSHAP": np.median(values, axis=0),
            "PositiveSHAPRate": (values > 0).mean(axis=0),
            "NegativeSHAPRate": (values < 0).mean(axis=0),
        }
    ).sort_values("MeanAbsSHAP", ascending=False)
    result["SHAPShare"] = result["MeanAbsSHAP"] / result["MeanAbsSHAP"].sum()
    if feature_info is not None:
        result = result.merge(feature_info, on="Feature", how="left")
    return result


def save_summary_plot(values, X: pd.DataFrame, path: str | Path, max_display: int = 30) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    shap.summary_plot(values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def save_bar_plot(importance: pd.DataFrame, path: str | Path, top_n: int = 30) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    top = importance.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, max(5, top_n * 0.25)))
    plt.barh(top["Feature"], top["MeanAbsSHAP"], color="#4C78A8")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()

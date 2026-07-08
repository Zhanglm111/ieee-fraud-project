from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def near_zero_variance_filter(X_train: pd.DataFrame, threshold: float = 0.0) -> tuple[list[str], VarianceThreshold]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    kept = X_train.columns[selector.get_support()].tolist()
    return kept, selector


def correlation_filter(X: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    if X.shape[1] <= 1:
        return X.columns.tolist()
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [col for col in X.columns if col not in drop]


def vif_filter(
    X: pd.DataFrame,
    threshold: float = 10.0,
    max_iter: int = 200,
    sample_size: int | None = 100_000,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    if threshold is None or threshold <= 0 or X.shape[1] <= 1:
        return X.columns.tolist(), pd.DataFrame(columns=["Step", "Feature", "VIF", "Action"])

    work = X.copy()
    if sample_size and len(work) > sample_size:
        work = work.sample(n=sample_size, random_state=random_state)
    work = work.astype(float)
    work = work.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    std = work.std(axis=0)
    variable_cols = std[std > 0].index.tolist()
    dropped_constant = [col for col in work.columns if col not in variable_cols]
    work = work[variable_cols]

    rows = [
        {"Step": 0, "Feature": col, "VIF": np.nan, "Action": "drop_constant"}
        for col in dropped_constant
    ]
    step = 1

    while work.shape[1] > 1 and step <= max_iter:
        corr = work.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        inv_corr = np.linalg.pinv(corr)
        vif_values = pd.Series(np.diag(inv_corr), index=work.columns)
        max_feature = str(vif_values.idxmax())
        max_vif = float(vif_values.loc[max_feature])
        if max_vif <= threshold:
            for feature, vif in vif_values.sort_values(ascending=False).items():
                rows.append({"Step": step, "Feature": feature, "VIF": float(vif), "Action": "keep_final"})
            break
        rows.append({"Step": step, "Feature": max_feature, "VIF": max_vif, "Action": "drop_high_vif"})
        work = work.drop(columns=[max_feature])
        step += 1

    return work.columns.tolist(), pd.DataFrame(rows)

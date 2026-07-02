from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def drop_near_zero_variance(X_train: pd.DataFrame, threshold: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train)
    kept = X_train.columns[selector.get_support()].tolist()
    return X_train[kept].copy(), kept


def correlation_filter(X: pd.DataFrame, threshold: float = 0.8) -> list[str]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [col for col in X.columns if col not in drop]

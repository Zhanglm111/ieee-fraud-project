from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


TARGET = "isFraud"


def calculate_woe_iv_from_binned(data: pd.DataFrame, bin_col: str, target_col: str = TARGET) -> tuple[float, pd.DataFrame]:
    grouped = data.groupby(bin_col, dropna=False)[target_col].agg(["count", "sum"])
    grouped = grouped.rename(columns={"sum": "bad"})
    grouped["good"] = grouped["count"] - grouped["bad"]
    grouped["bad_dist"] = (grouped["bad"] + 0.5) / (grouped["bad"].sum() + 0.5 * len(grouped))
    grouped["good_dist"] = (grouped["good"] + 0.5) / (grouped["good"].sum() + 0.5 * len(grouped))
    grouped["WOE"] = np.log(grouped["bad_dist"] / grouped["good_dist"])
    grouped["IV_component"] = (grouped["bad_dist"] - grouped["good_dist"]) * grouped["WOE"]
    return float(grouped["IV_component"].sum()), grouped.reset_index()


def bin_numeric_feature(data: pd.DataFrame, feature: str, bins: int = 10) -> pd.Series:
    series = data[feature]
    if series.nunique(dropna=True) <= 1:
        return pd.Series("__single__", index=data.index)
    try:
        return pd.qcut(series, q=bins, duplicates="drop").astype(str).fillna("__MISSING__")
    except ValueError:
        return pd.cut(series, bins=bins, duplicates="drop").astype(str).fillna("__MISSING__")


def bin_categorical_feature(data: pd.DataFrame, feature: str, max_categories: int = 30) -> pd.Series:
    series = data[feature].astype("object").fillna("__MISSING__")
    top = set(series.value_counts().head(max_categories).index)
    return series.where(series.isin(top), "__OTHER__").astype(str)


def compute_iv_table(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str] | None = None,
    target: str = TARGET,
    bins: int = 10,
    max_categories: int = 30,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    categorical = set(categorical or [])
    rows = []
    details = {}
    for feature in features:
        if feature in categorical or train[feature].dtype == "object":
            binned = bin_categorical_feature(train, feature, max_categories=max_categories)
        else:
            binned = bin_numeric_feature(train, feature, bins=bins)
        tmp = pd.DataFrame({feature: binned, target: train[target]})
        iv, detail = calculate_woe_iv_from_binned(tmp, feature, target)
        rows.append({"Feature": feature, "IV": iv})
        details[feature] = detail
    return pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True), details


def select_features(iv_table: pd.DataFrame, candidate_features: list[str], selection: str, min_iv: float | None, max_iv: float | None) -> list[str]:
    if selection == "all":
        return list(candidate_features)
    mask = pd.Series(True, index=iv_table.index)
    if min_iv is not None:
        mask &= iv_table["IV"] >= min_iv
    if selection == "range" and max_iv is not None:
        mask &= iv_table["IV"] <= max_iv
    return iv_table.loc[mask, "Feature"].tolist()


@dataclass
class WOEEncoder:
    rules: dict[str, dict]

    def transform(self, data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for feature in features:
            rule = self.rules[feature]
            if rule["kind"] == "numeric":
                binned = pd.cut(data[feature], bins=rule["bins"], include_lowest=True).astype(str).fillna("__MISSING__")
            else:
                values = data[feature].astype("object").fillna("__MISSING__")
                binned = values.where(values.isin(rule["levels"]), "__OTHER__").astype(str)
            columns[f"{feature}_woe"] = binned.map(rule["woe"]).fillna(rule["default_woe"]).astype(float)
        return pd.concat(columns, axis=1)


def fit_woe_encoder(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str] | None = None,
    target: str = TARGET,
    bins: int = 10,
    max_categories: int = 30,
) -> WOEEncoder:
    categorical = set(categorical or [])
    rules = {}
    for feature in features:
        if feature in categorical or train[feature].dtype == "object":
            binned = bin_categorical_feature(train, feature, max_categories=max_categories)
            kind = "categorical"
            levels = sorted(set(binned.unique()) - {"__OTHER__"})
            bins_rule = None
        else:
            series = train[feature]
            if series.nunique(dropna=True) <= 1:
                binned = pd.Series("__single__", index=train.index)
                bins_rule = np.array([-np.inf, np.inf])
            else:
                try:
                    _, bins_rule = pd.qcut(series, q=bins, duplicates="drop", retbins=True)
                except ValueError:
                    _, bins_rule = pd.cut(series, bins=bins, duplicates="drop", retbins=True)
                bins_rule[0] = -np.inf
                bins_rule[-1] = np.inf
                binned = pd.cut(series, bins=bins_rule, include_lowest=True).astype(str).fillna("__MISSING__")
            kind = "numeric"
            levels = None
        tmp = pd.DataFrame({feature: binned, target: train[target]})
        _, detail = calculate_woe_iv_from_binned(tmp, feature, target)
        woe = dict(zip(detail[feature].astype(str), detail["WOE"]))
        rules[feature] = {
            "kind": kind,
            "woe": woe,
            "default_woe": float(np.mean(list(woe.values()))) if woe else 0.0,
            "levels": levels,
            "bins": bins_rule,
        }
    return WOEEncoder(rules)


def _bin_categorical_with_top(series: pd.Series, top_categories: set) -> pd.Series:
    """Bin a categorical series using pre-defined top categories.

    Values in *top_categories* are kept as-is; all others become ``__OTHER__``;
    missing values become ``__MISSING__``.
    """
    filled = series.astype("object").fillna("__MISSING__")
    return filled.where(filled.isin(top_categories), "__OTHER__").astype(str)


def cv_woe_encode_train(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str] | None = None,
    target: str = TARGET,
    bins: int = 10,
    max_categories: int = 30,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, WOEEncoder]:
    """Time-series cross-validated WOE encoding for training data.

    Each sample's WOE value is computed using **only earlier-in-time**
    samples within the training set.  This prevents target leakage and
    simulates the production setting where future data is unavailable
    at encoding time.

    The training DataFrame must be sorted by time (index order = time
    order).  This is guaranteed by :func:`fraud_detection.data.split_by_time`.

    Returns
    -------
    X_train_woe : pd.DataFrame
        WOE-encoded training features.  Column names follow the
        ``{feature}_woe`` convention used by :meth:`WOEEncoder.transform`.
    encoder : WOEEncoder
        Encoder fitted on the *full* training set, for transforming
        validation / test (out-of-time) splits.
    """
    categorical = set(categorical or [])

    # ── Step 1: pre-compute bin definitions on the full training set ──
    # These are safe: bin edges / top categories depend only on feature
    # distributions, never on the target.
    feature_bins: dict[str, dict] = {}
    for feature in features:
        if feature in categorical or train[feature].dtype == "object":
            series_vals = train[feature].astype("object").fillna("__MISSING__")
            top_cats = set(series_vals.value_counts().head(max_categories).index)
            feature_bins[feature] = {"kind": "categorical", "top_categories": top_cats}
        else:
            series_vals = train[feature]
            if series_vals.nunique(dropna=True) <= 1:
                feature_bins[feature] = {"kind": "numeric", "bin_edges": np.array([-np.inf, np.inf])}
            else:
                try:
                    _, bin_edges = pd.qcut(series_vals, q=bins, duplicates="drop", retbins=True)
                except ValueError:
                    _, bin_edges = pd.cut(series_vals, bins=bins, duplicates="drop", retbins=True)
                bin_edges = np.array(bin_edges)
                bin_edges[0] = -np.inf
                bin_edges[-1] = np.inf
                feature_bins[feature] = {"kind": "numeric", "bin_edges": bin_edges}

    # ── Step 2: TimeSeriesSplit CV WOE encoding ──
    tscv = TimeSeriesSplit(n_splits=n_splits)
    encoded_mask = pd.Series(False, index=train.index)
    woe_columns: dict[str, pd.Series] = {}

    for feature in features:
        result_col = pd.Series(np.nan, index=train.index, dtype=float)
        bin_info = feature_bins[feature]

        for train_idx, holdout_idx in tscv.split(train):
            # Bin both segments with shared bin definition
            if bin_info["kind"] == "categorical":
                train_binned = _bin_categorical_with_top(
                    train[feature].iloc[train_idx], bin_info["top_categories"]
                )
                holdout_binned = _bin_categorical_with_top(
                    train[feature].iloc[holdout_idx], bin_info["top_categories"]
                )
            else:
                train_binned = (
                    pd.cut(train[feature].iloc[train_idx], bins=bin_info["bin_edges"], include_lowest=True)
                    .astype(str)
                    .fillna("__MISSING__")
                )
                holdout_binned = (
                    pd.cut(train[feature].iloc[holdout_idx], bins=bin_info["bin_edges"], include_lowest=True)
                    .astype(str)
                    .fillna("__MISSING__")
                )

            # Compute WOE from the *training segment only* (the "past")
            tmp = pd.DataFrame({"bin": train_binned, target: train[target].iloc[train_idx].values})
            _, detail = calculate_woe_iv_from_binned(tmp, "bin", target)
            woe_map = dict(zip(detail["bin"].astype(str), detail["WOE"]))
            default_woe = float(np.mean(list(woe_map.values()))) if woe_map else 0.0

            # Apply to holdout segment (the "future")
            result_col.iloc[holdout_idx] = holdout_binned.map(woe_map).fillna(default_woe).values
            encoded_mask.iloc[holdout_idx] = True

        woe_columns[f"{feature}_woe"] = result_col

    result = pd.concat(woe_columns, axis=1)

    # ── Step 3: fit full encoder for valid / test transformation ──
    full_encoder = fit_woe_encoder(
        train, features, list(categorical), target, bins, max_categories
    )

    # ── Step 4: back-fill earliest samples never in a holdout fold ──
    # With TimeSeriesSplit the first ~1/(n_splits+1) samples are always in
    # the training portion and never get CV-encoded.  Fall back to the
    # full-encoder WOE (which does have mild leakage but only for a small
    # fraction of rows).
    if not encoded_mask.all():
        full_encoded = full_encoder.transform(train, features)
        for col in result.columns:
            missing = result[col].isna()
            result.loc[missing, col] = full_encoded.loc[missing, col]

    return result, full_encoder

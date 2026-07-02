from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    return pd.DataFrame(rows).sort_values("IV", ascending=False), details


def select_features(iv_table: pd.DataFrame, selection: str, min_iv: float | None, max_iv: float | None) -> list[str]:
    if selection == "all":
        return iv_table["Feature"].tolist()
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
        result = pd.DataFrame(index=data.index)
        for feature in features:
            rule = self.rules[feature]
            if rule["kind"] == "numeric":
                binned = pd.cut(data[feature], bins=rule["bins"], include_lowest=True).astype(str)
            else:
                values = data[feature].astype("object").fillna("__MISSING__")
                binned = values.where(values.isin(rule["levels"]), "__OTHER__").astype(str)
            result[f"{feature}_woe"] = binned.map(rule["woe"]).fillna(rule["default_woe"]).astype(float)
        return result


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
            _, bins_edges = pd.qcut(train[feature], q=bins, duplicates="drop", retbins=True)
            binned = pd.cut(train[feature], bins=bins_edges, include_lowest=True).astype(str).fillna("__MISSING__")
            kind = "numeric"
            levels = None
            bins_rule = bins_edges
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

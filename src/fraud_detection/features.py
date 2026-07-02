from __future__ import annotations

import re

import numpy as np
import pandas as pd


TARGET = "isFraud"


def get_base_feature_name(feature: str) -> str:
    for suffix in ("_is_missing", "_freq", "_woe"):
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def infer_feature_category(feature: str) -> str:
    base = get_base_feature_name(feature)
    if base in {"TransactionDT", "relative_day", "hour", "TransactionAmt", "TransactionAmt_log", "ProductCD"}:
        return "transaction_context"
    if re.match(r"^card[1-6]$", base) or base in {"addr1", "addr2", "P_emaildomain", "R_emaildomain"}:
        return "payment_identity"
    if re.match(r"^C\d+$", base):
        return "count_aggregation"
    if re.match(r"^D\d+$", base):
        return "time_delta"
    if re.match(r"^M\d+$", base):
        return "identity_match"
    if re.match(r"^V\d+$", base):
        return "vesta_aggregation"
    if base.startswith("id_") or base in {"DeviceType", "DeviceInfo"}:
        return "device_fingerprint"
    if base in {"dist1", "dist2"}:
        return "behavior_drift"
    return "other"


def infer_risk_mechanism(feature: str) -> str:
    category = infer_feature_category(feature)
    return {
        "transaction_context": "transaction_context_anomaly",
        "payment_identity": "payment_identity_anomaly",
        "count_aggregation": "entity_aggregation_anomaly",
        "vesta_aggregation": "entity_aggregation_anomaly",
        "time_delta": "behavior_timing_anomaly",
        "identity_match": "identity_consistency_anomaly",
        "device_fingerprint": "device_network_fingerprint_anomaly",
        "behavior_drift": "behavior_drift_anomaly",
    }.get(category, "other_risk_signal")


def build_feature_system(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    rows = []
    for feature in df.columns:
        if feature in {target, "TransactionID"}:
            continue
        rows.append(
            {
                "Feature": feature,
                "FeatureCategory": infer_feature_category(feature),
                "RiskMechanism": infer_risk_mechanism(feature),
                "MissingRate": float(df[feature].isna().mean()),
                "DType": str(df[feature].dtype),
            }
        )
    return pd.DataFrame(rows)


def candidate_features(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    excluded = {target, "TransactionID"}
    return [col for col in df.columns if col not in excluded]


def add_missing_indicators(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train = train.copy()
    valid = valid.copy()
    test = test.copy()
    missing_rate = train[features].isna().mean()
    high_missing = missing_rate[missing_rate > threshold].index.tolist()
    indicators = []
    for col in high_missing:
        name = f"{col}_is_missing"
        train[name] = train[col].isna().astype("int8")
        valid[name] = valid[col].isna().astype("int8")
        test[name] = test[col].isna().astype("int8")
        indicators.append(name)
    normal_features = [col for col in features if col not in high_missing]
    return train, valid, test, normal_features + indicators, indicators


def categorical_features(features: list[str]) -> list[str]:
    cats = []
    for col in features:
        base = get_base_feature_name(col)
        if (
            base in {"ProductCD", "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"}
            or re.match(r"^card[1-6]$", base)
            or re.match(r"^M\d+$", base)
            or base.startswith("id_")
        ):
            cats.append(col)
    return cats


def prepare_tree_matrices(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str = TARGET,
    force_categorical: list[str] | None = None,
) -> dict[str, pd.DataFrame | pd.Series | pd.DataFrame]:
    force_categorical = set(force_categorical or [])
    X_train = train[features].copy()
    X_valid = valid[features].copy()
    X_test = test[features].copy()
    y_train = train[target].astype(int)
    y_valid = valid[target].astype(int)
    y_test = test[target].astype(int)

    category_mappings: dict[str, dict] = {}
    for col in features:
        if col in force_categorical or X_train[col].dtype == "object":
            X_train[col] = X_train[col].astype("object").fillna("__MISSING__")
            X_valid[col] = X_valid[col].astype("object").fillna("__MISSING__")
            X_test[col] = X_test[col].astype("object").fillna("__MISSING__")
            levels = pd.Index(X_train[col].unique())
            mapping = {value: i for i, value in enumerate(levels)}
            category_mappings[col] = mapping
            X_train[col] = X_train[col].map(mapping).fillna(-1).astype("int32")
            X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype("int32")
            X_test[col] = X_test[col].map(mapping).fillna(-1).astype("int32")
        else:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_valid[col] = X_valid[col].fillna(median)
            X_test[col] = X_test[col].fillna(median)

    feature_info = pd.DataFrame(
        {
            "Feature": features,
            "BaseFeature": [get_base_feature_name(col) for col in features],
            "FeatureCategory": [infer_feature_category(col) for col in features],
            "RiskMechanism": [infer_risk_mechanism(col) for col in features],
            "FeatureType": [
                "categorical_encoded" if col in category_mappings else "numeric" for col in features
            ],
        }
    )
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "feature_info": feature_info,
        "category_mappings": category_mappings,
    }

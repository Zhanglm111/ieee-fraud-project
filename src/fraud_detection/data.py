from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, resolve_path


TARGET = "isFraud"


def read_ieee_train(raw_dir: str | Path, id_column: str = "TransactionID") -> pd.DataFrame:
    raw_dir = resolve_path(raw_dir)
    transaction = pd.read_csv(raw_dir / "train_transaction.csv")
    identity_path = raw_dir / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        transaction = transaction.merge(identity, how="left", on=id_column)
    return add_basic_columns(transaction)


def read_ieee_official_test(raw_dir: str | Path, id_column: str = "TransactionID") -> pd.DataFrame:
    raw_dir = resolve_path(raw_dir)
    transaction = pd.read_csv(raw_dir / "test_transaction.csv")
    identity_path = raw_dir / "test_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path)
        transaction = transaction.merge(identity, how="left", on=id_column)
    return add_basic_columns(transaction)


def add_basic_columns(df: pd.DataFrame, time_column: str = "TransactionDT") -> pd.DataFrame:
    df = df.copy()
    if time_column in df.columns:
        df["relative_day"] = (df[time_column] // 86400).astype("int32")
        df["hour"] = ((df[time_column] / 3600) % 24).astype("int16")
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    return df


def split_by_time(
    df: pd.DataFrame,
    time_column: str = "TransactionDT",
    train_size: float = 0.70,
    valid_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(time_column).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))
    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()
    profile = pd.DataFrame(
        [
            {
                "Split": "train",
                "Rows": len(train),
                "FraudRate": train[TARGET].mean() if TARGET in train else np.nan,
                "MinTime": train[time_column].min(),
                "MaxTime": train[time_column].max(),
            },
            {
                "Split": "valid",
                "Rows": len(valid),
                "FraudRate": valid[TARGET].mean() if TARGET in valid else np.nan,
                "MinTime": valid[time_column].min(),
                "MaxTime": valid[time_column].max(),
            },
            {
                "Split": "test",
                "Rows": len(test),
                "FraudRate": test[TARGET].mean() if TARGET in test else np.nan,
                "MinTime": test[time_column].min(),
                "MaxTime": test[time_column].max(),
            },
        ]
    )
    return train, valid, test, profile


def write_processed_splits(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    processed_dir: str | Path,
    stem: str,
) -> dict[str, str]:
    processed_dir = ensure_dir(resolve_path(processed_dir))
    paths = {
        "train": processed_dir / f"{stem}_train.parquet",
        "valid": processed_dir / f"{stem}_valid.parquet",
        "test": processed_dir / f"{stem}_test.parquet",
    }
    train.to_parquet(paths["train"], index=False)
    valid.to_parquet(paths["valid"], index=False)
    test.to_parquet(paths["test"], index=False)
    return {key: str(value) for key, value in paths.items()}


def prepare_data(config: dict) -> dict[str, str]:
    data_cfg = config["data"]
    df = read_ieee_train(data_cfg["raw_dir"], data_cfg.get("id_column", "TransactionID"))
    train, valid, test, profile = split_by_time(
        df,
        data_cfg.get("time_column", "TransactionDT"),
        data_cfg.get("train_size", 0.70),
        data_cfg.get("valid_size", 0.15),
    )
    paths = write_processed_splits(
        train,
        valid,
        test,
        data_cfg["processed_dir"],
        data_cfg.get("output_name", "ieee_train_time_split"),
    )
    profile_path = resolve_path(data_cfg["processed_dir"]) / f"{data_cfg.get('output_name', 'ieee_train_time_split')}_profile.csv"
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")
    paths["profile"] = str(profile_path)
    return paths

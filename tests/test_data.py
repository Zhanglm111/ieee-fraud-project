import pandas as pd

from fraud_detection.data import add_basic_columns, split_by_time


def test_add_basic_columns():
    df = pd.DataFrame({"TransactionDT": [86400, 90000], "TransactionAmt": [10.0, 20.0]})
    out = add_basic_columns(df)
    assert {"relative_day", "hour", "TransactionAmt_log"}.issubset(out.columns)


def test_split_by_time():
    df = pd.DataFrame({"TransactionDT": range(100), "isFraud": [0] * 100})
    train, valid, test, profile = split_by_time(df)
    assert len(train) == 70
    assert len(valid) == 15
    assert len(test) == 15
    assert len(profile) == 3

import pandas as pd

from fraud_detection.iv import compute_iv_table


def test_compute_iv_table():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3], "isFraud": [0, 0, 0, 1, 1, 1]})
    iv, details = compute_iv_table(df, ["x"])
    assert iv.shape[0] == 1
    assert "x" in details

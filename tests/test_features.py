import pandas as pd

from fraud_detection.features import candidate_features, categorical_features, infer_feature_category, infer_risk_mechanism


def test_feature_category_mapping():
    assert infer_feature_category("C13") == "行为计数统计特征"
    assert infer_feature_category("V294") == "Vesta聚合统计特征"
    assert infer_feature_category("dist1") == "支付工具与身份代理特征"
    assert infer_risk_mechanism("D2") == "行为时序节奏异常"
    assert infer_risk_mechanism("dist1") == "支付身份与联系方式异常"


def test_candidate_features_follow_data_dictionary():
    df = pd.DataFrame(
        columns=[
            "TransactionID",
            "isFraud",
            "TransactionDT",
            "TransactionAmt",
            "relative_day",
            "hour",
            "TransactionAmt_log",
            "ProductCD",
            "card1",
            "id_01",
            "id_12",
            "DeviceType",
        ]
    )
    features = candidate_features(df)
    assert "hour" in features
    assert "TransactionAmt_log" in features
    assert "ProductCD" in features
    assert "TransactionDT" not in features
    assert "TransactionAmt" not in features
    assert "relative_day" not in features


def test_identity_field_types_follow_data_dictionary():
    cats = categorical_features(["id_01", "id_02", "id_11", "id_12", "id_38", "DeviceType"])
    assert "id_01" not in cats
    assert "id_02" not in cats
    assert "id_11" not in cats
    assert "id_12" in cats
    assert "id_38" in cats
    assert "DeviceType" in cats

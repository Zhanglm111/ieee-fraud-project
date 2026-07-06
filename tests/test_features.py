from fraud_detection.features import infer_feature_category, infer_risk_mechanism


def test_feature_category_mapping():
    assert infer_feature_category("C13") == "行为计数统计特征"
    assert infer_feature_category("V294") == "Vesta聚合统计特征"
    assert infer_risk_mechanism("D2") == "行为时序节奏异常"

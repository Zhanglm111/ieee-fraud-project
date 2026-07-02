from fraud_detection.features import infer_feature_category, infer_risk_mechanism


def test_feature_category_mapping():
    assert infer_feature_category("C13") == "count_aggregation"
    assert infer_feature_category("V294") == "vesta_aggregation"
    assert infer_risk_mechanism("D2") == "behavior_timing_anomaly"

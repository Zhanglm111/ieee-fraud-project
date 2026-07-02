import numpy as np

from fraud_detection.evaluate import classification_metrics, topk_evaluation


def test_metrics():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = classification_metrics(y, score, 0.5)
    assert metrics["AUC"] == 1.0
    assert len(topk_evaluation(y, score, [0.5])) == 1

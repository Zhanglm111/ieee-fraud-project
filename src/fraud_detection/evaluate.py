from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_ks(y_true, y_score) -> float:
    data = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    total_bad = data["y"].sum()
    total_good = len(data) - total_bad
    bad_cum = data["y"].cumsum() / max(total_bad, 1)
    good_cum = (1 - data["y"]).cumsum() / max(total_good, 1)
    return float((bad_cum - good_cum).abs().max())


def find_best_threshold(y_true, y_score) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def classification_metrics(y_true, y_score, threshold: float) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "KS": calculate_ks(y_true, y_score),
        "Threshold": float(threshold),
    }


def topk_evaluation(y_true, y_score, rates=(0.01, 0.03, 0.05, 0.10)) -> pd.DataFrame:
    data = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    total_bad = data["y"].sum()
    base_rate = data["y"].mean()
    rows = []
    for rate in rates:
        k = max(1, int(len(data) * rate))
        top = data.head(k)
        precision = top["y"].mean()
        recall_capture = top["y"].sum() / max(total_bad, 1)
        rows.append(
            {
                "TopRate": rate,
                "TopN": k,
                "Precision": float(precision),
                "RecallCapture": float(recall_capture),
                "Lift": float(precision / base_rate) if base_rate > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_frame(y_true, y_score, threshold: float) -> pd.DataFrame:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame([{"TN": tn, "FP": fp, "FN": fn, "TP": tp}])

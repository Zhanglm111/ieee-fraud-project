from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data as data_mod
from . import explain as explain_mod
from . import features as feature_mod
from . import iv as iv_mod
from . import models as model_mod
from .evaluate import classification_metrics, find_best_threshold, topk_evaluation
from .utils import ensure_dir, save_json


def load_processed_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    processed_dir = Path(data_cfg["processed_dir"])
    stem = data_cfg.get("output_name", "ieee_train_time_split")
    paths = [
        processed_dir / f"{stem}_train.parquet",
        processed_dir / f"{stem}_valid.parquet",
        processed_dir / f"{stem}_test.parquet",
    ]
    if not all(path.exists() for path in paths):
        data_mod.prepare_data(config)
    return tuple(pd.read_parquet(path) for path in paths)


def run_training(config: dict, output_dir: str | Path = "outputs") -> None:
    output_dir = ensure_dir(output_dir)
    train, valid, test = load_processed_splits(config)
    target = config["project"].get("target", "isFraud")
    all_features = feature_mod.candidate_features(train, target)
    train, valid, test, iv_features, indicators = feature_mod.add_missing_indicators(
        train,
        valid,
        test,
        all_features,
        config["features"].get("high_missing_threshold", 0.95),
    )
    categories = feature_mod.categorical_features(iv_features)
    iv_table, _ = iv_mod.compute_iv_table(
        train,
        iv_features,
        categories,
        target=target,
        bins=config["iv"].get("bins", 10),
    )
    iv_table.to_csv(output_dir / "iv" / "iv_summary.csv", index=False, encoding="utf-8-sig")

    for scheme in config["training"]["schemes"]:
        scheme_dir = ensure_dir(output_dir / scheme["name"])
        selected = feature_mod.candidate_features(train, target) if scheme["selection"] == "all" else iv_mod.select_features(
            iv_table, scheme["selection"], scheme.get("min_iv"), scheme.get("max_iv")
        )
        pd.DataFrame({"Feature": selected}).to_csv(
            scheme_dir / "selected_features.csv", index=False, encoding="utf-8-sig"
        )
        tree = feature_mod.prepare_tree_matrices(train, valid, test, selected, target, categories)
        tree["feature_info"].to_csv(scheme_dir / "feature_info.csv", index=False, encoding="utf-8-sig")
        model_name = config["model"]["name"]
        params = config["model"].get("params", {})
        model = model_mod.make_model(model_name, params, config["project"].get("random_state", 42))
        if model_name == "xgboost":
            model.fit(
                tree["X_train"],
                tree["y_train"],
                eval_set=[(tree["X_valid"], tree["y_valid"])],
                verbose=False,
            )
        else:
            model.fit(tree["X_train"], tree["y_train"])
        model_mod.save_model(model, scheme_dir / "model.joblib")
        save_json({"scheme": scheme, "model": model_name, "selected_feature_count": len(selected)}, scheme_dir / "run_config.json")


def run_evaluation(config: dict, experiment: str, output_dir: str | Path = "outputs") -> None:
    output_dir = Path(output_dir)
    train, valid, test = load_processed_splits(config)
    target = config["project"].get("target", "isFraud")
    selected = pd.read_csv(output_dir / experiment / "selected_features.csv")["Feature"].tolist()
    categories = feature_mod.categorical_features(selected)
    tree = feature_mod.prepare_tree_matrices(train, valid, test, selected, target, categories)
    model = model_mod.load_model(output_dir / experiment / "model.joblib")
    valid_score = model.predict_proba(tree["X_valid"])[:, 1]
    threshold = find_best_threshold(tree["y_valid"], valid_score)
    rows = []
    for split, X, y in [
        ("train", tree["X_train"], tree["y_train"]),
        ("valid", tree["X_valid"], tree["y_valid"]),
        ("test", tree["X_test"], tree["y_test"]),
    ]:
        score = model.predict_proba(X)[:, 1]
        metric = classification_metrics(y, score, threshold)
        metric.update({"Split": split, "Experiment": experiment})
        rows.append(metric)
        topk_evaluation(y, score, config["evaluation"].get("top_rates", [0.01, 0.03, 0.05, 0.10])).to_csv(
            output_dir / experiment / f"topk_{split}.csv", index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(rows).to_csv(output_dir / experiment / "metrics.csv", index=False, encoding="utf-8-sig")


def run_explain(config: dict, experiment: str, output_dir: str | Path = "outputs") -> None:
    output_dir = Path(output_dir)
    train, valid, test = load_processed_splits(config)
    target = config["project"].get("target", "isFraud")
    selected = pd.read_csv(output_dir / experiment / "selected_features.csv")["Feature"].tolist()
    categories = feature_mod.categorical_features(selected)
    tree = feature_mod.prepare_tree_matrices(train, valid, test, selected, target, categories)
    model = model_mod.load_model(output_dir / experiment / "model.joblib")
    shap_dir = ensure_dir(output_dir / experiment / "shap")
    sample = explain_mod.sample_for_shap(
        tree["X_test"],
        config["explain"].get("sample_size", 5000),
        config["project"].get("random_state", 42),
        output_dir / "shap_sample_indices.csv",
    )
    values = explain_mod.compute_tree_shap(model, sample)
    importance = explain_mod.shap_importance_table(sample, values, tree["feature_info"])
    importance.to_csv(shap_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
    explain_mod.save_bar_plot(importance, shap_dir / "shap_importance_top30.png", config["explain"].get("max_display", 30))
    explain_mod.save_summary_plot(values, sample, shap_dir / "shap_summary_top30.png", config["explain"].get("max_display", 30))

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pandas as pd

from . import data as data_mod
from . import explain as explain_mod
from . import features as feature_mod
from . import iv as iv_mod
from . import models as model_mod
from .evaluate import classification_metrics, find_best_threshold, topk_evaluation
from .preprocessing import correlation_filter, near_zero_variance_filter, vif_filter
from .utils import ensure_dir, resolve_path, save_json


def configure_training_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fraud_detection.training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "logs" / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_processed_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    processed_dir = resolve_path(data_cfg["processed_dir"])
    stem = data_cfg.get("output_name", "ieee_train_time_split")
    paths = [
        processed_dir / f"{stem}_train.parquet",
        processed_dir / f"{stem}_valid.parquet",
        processed_dir / f"{stem}_test.parquet",
    ]
    if not all(path.exists() for path in paths):
        data_mod.prepare_data(config)
    return tuple(pd.read_parquet(path) for path in paths)


def enabled_models(config: dict) -> dict:
    return {name: cfg for name, cfg in config.get("models", {}).items() if cfg.get("enabled", False)}


def prepare_base_features(config: dict):
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
    force_categorical = feature_mod.categorical_features(iv_features)
    risk_system = feature_mod.build_feature_system(train, target, iv_features)
    return train, valid, test, iv_features, force_categorical, risk_system


def prepare_woe_lr_matrices(config: dict, train, valid, test, selected_features, force_categorical):
    target = config["project"].get("target", "isFraud")
    encoder = iv_mod.fit_woe_encoder(
        train,
        selected_features,
        [f for f in force_categorical if f in selected_features],
        target=target,
        bins=config["iv"].get("bins", 10),
        max_categories=config["iv"].get("max_categories", 30),
    )
    X_train = encoder.transform(train, selected_features)
    X_valid = encoder.transform(valid, selected_features)
    X_test = encoder.transform(test, selected_features)
    model_cfg = config["models"].get("woe_lr", {})
    prep_cfg = model_cfg.get("preprocessing", {})
    filter_rows = []
    initial_features = list(X_train.columns)

    kept, selector = near_zero_variance_filter(X_train, prep_cfg.get("near_zero_variance_threshold", 0.0))
    variance_drop = [col for col in X_train.columns if col not in kept]
    filter_rows.extend({"Step": "near_zero_variance", "Feature": col, "Action": "drop"} for col in variance_drop)
    X_train = X_train[kept]
    X_valid = X_valid[kept]
    X_test = X_test[kept]

    kept = correlation_filter(X_train, prep_cfg.get("correlation_threshold", 0.8))
    corr_drop = [col for col in X_train.columns if col not in kept]
    filter_rows.extend({"Step": "correlation", "Feature": col, "Action": "drop"} for col in corr_drop)
    X_train = X_train[kept]
    X_valid = X_valid[kept]
    X_test = X_test[kept]

    vif_report = pd.DataFrame()
    if prep_cfg.get("vif_enabled", True):
        kept, vif_report = vif_filter(
            X_train,
            threshold=prep_cfg.get("vif_threshold", 10.0),
            max_iter=prep_cfg.get("vif_max_iter", 200),
            sample_size=prep_cfg.get("vif_sample_size", 100_000),
            random_state=config["project"].get("random_state", 42),
        )
        vif_drop = [col for col in X_train.columns if col not in kept]
        filter_rows.extend({"Step": "vif", "Feature": col, "Action": "drop"} for col in vif_drop)
        X_train = X_train[kept]
        X_valid = X_valid[kept]
        X_test = X_test[kept]

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": train[target].astype(int),
        "y_valid": valid[target].astype(int),
        "y_test": test[target].astype(int),
        "encoder": encoder,
        "features": kept,
        "filter_summary": {
            "initial_features": len(initial_features),
            "near_zero_variance_drop_features": len(variance_drop),
            "correlation_drop_features": len(corr_drop),
            "vif_drop_features": len([row for row in filter_rows if row["Step"] == "vif"]),
            "final_features": len(kept),
        },
        "filter_detail": pd.DataFrame(filter_rows),
        "vif_report": vif_report,
    }


def predict_score(model, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def evaluate_model(model, scheme_name: str, model_name: str, matrices: dict, threshold: float, top_rates):
    metrics_rows = []
    topk_rows = []
    prediction_frames = []
    for split, X, y in [
        ("Train", matrices["X_train"], matrices["y_train"]),
        ("Valid", matrices["X_valid"], matrices["y_valid"]),
        ("Test", matrices["X_test"], matrices["y_test"]),
    ]:
        score = predict_score(model, X)
        metrics = classification_metrics(y, score, threshold)
        metrics.update({"Scheme": scheme_name, "Model": model_name, "Split": split})
        metrics_rows.append(metrics)
        topk = topk_evaluation(y, score, top_rates)
        topk["Scheme"] = scheme_name
        topk["Model"] = model_name
        topk["Split"] = split
        topk_rows.append(topk)
        prediction_frames.append(
            pd.DataFrame({"Scheme": scheme_name, "Model": model_name, "Split": split, "y_true": y, "score": score})
        )
    return metrics_rows, pd.concat(topk_rows, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)


def run_training(config: dict, output_dir: str | Path = "outputs") -> None:
    output_dir = ensure_dir(resolve_path(output_dir))
    for sub in ["tables", "iv", "models", "metrics", "predictions", "logs"]:
        ensure_dir(output_dir / sub)
    logger = configure_training_logger(output_dir)
    logger.info("Training started. output_dir=%s", output_dir)
    logger.info("Preparing data splits and base feature system.")
    train, valid, test, iv_features, force_categorical, risk_system = prepare_base_features(config)
    target = config["project"].get("target", "isFraud")
    logger.info(
        "Data ready. train=%d, valid=%d, test=%d, candidate_features=%d, categorical_features=%d",
        len(train),
        len(valid),
        len(test),
        len(iv_features),
        len(force_categorical),
    )
    risk_system.to_csv(output_dir / "tables" / "risk_feature_system.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved risk feature system: %s", output_dir / "tables" / "risk_feature_system.csv")
    logger.info("Computing IV table.")
    iv_table, iv_details = iv_mod.compute_iv_table(
        train,
        iv_features,
        force_categorical,
        target=target,
        bins=config["iv"].get("bins", 10),
        max_categories=config["iv"].get("max_categories", 30),
    )
    iv_table = iv_table.merge(
        risk_system[["Feature", "FeatureCategory", "RiskMechanism"]],
        on="Feature",
        how="left",
    )
    iv_table.to_csv(output_dir / "iv" / "iv_summary.csv", index=False, encoding="utf-8-sig")
    iv_table.to_csv(output_dir / "tables" / "iv_summary.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved IV table with %d rows.", len(iv_table))

    all_metric_rows = []
    all_topk = []
    all_predictions = []
    scheme_rows = []
    for scheme in config["training"]["schemes"]:
        scheme_name = scheme["name"]
        logger.info("Scheme started: %s", scheme_name)
        scheme_dir = ensure_dir(output_dir / scheme_name)
        for sub in ["models", "tables", "metrics", "predictions", "shap"]:
            ensure_dir(scheme_dir / sub)
        selected = iv_mod.select_features(
            iv_table,
            iv_features,
            scheme["selection"],
            scheme.get("min_iv"),
            scheme.get("max_iv"),
        )
        logger.info("Scheme %s selected %d features.", scheme_name, len(selected))
        pd.DataFrame({"Feature": selected}).to_csv(scheme_dir / "tables" / "selected_features.csv", index=False, encoding="utf-8-sig")
        logger.info("Scheme %s preparing tree model matrices.", scheme_name)
        tree = feature_mod.prepare_tree_matrices(train, valid, test, selected, target, [f for f in force_categorical if f in selected])
        tree["feature_info"].to_csv(scheme_dir / "tables" / "model_feature_info.csv", index=False, encoding="utf-8-sig")
        model_mod.save_model(
            {
                "fill_values": tree["fill_values"],
                "category_mappings": tree["category_mappings"],
                "feature_info": tree["feature_info"],
            },
            scheme_dir / "models" / "tree_preprocess.joblib",
        )
        woe_data = None
        if config["models"].get("woe_lr", {}).get("enabled", False):
            logger.info("Scheme %s preparing WOE-LR matrices.", scheme_name)
            woe_data = prepare_woe_lr_matrices(config, train, valid, test, selected, force_categorical)
            woe_feature_info = pd.DataFrame(
                {
                    "Feature": woe_data["features"],
                    "BaseFeature": [feature_mod.get_base_feature_name(col) for col in woe_data["features"]],
                    "FeatureCategory": [feature_mod.infer_feature_category(col) for col in woe_data["features"]],
                    "RiskMechanism": [feature_mod.infer_risk_mechanism(col) for col in woe_data["features"]],
                }
            )
            woe_feature_info.to_csv(
                scheme_dir / "tables" / "woe_feature_info.csv", index=False, encoding="utf-8-sig"
            )
            woe_data["filter_detail"].to_csv(
                scheme_dir / "tables" / "woe_filter_detail.csv", index=False, encoding="utf-8-sig"
            )
            woe_data["vif_report"].to_csv(
                scheme_dir / "tables" / "woe_vif_report.csv", index=False, encoding="utf-8-sig"
            )
            model_mod.save_model(
                {
                    "encoder": woe_data["encoder"],
                    "features": woe_data["features"],
                    "filter_summary": woe_data["filter_summary"],
                },
                scheme_dir / "models" / "woe_preprocess.joblib",
            )
            save_json(woe_data["filter_summary"], scheme_dir / "tables" / "woe_filter_summary.json")

        scheme_rows.append(
            {
                "Scheme": scheme_name,
                "Label": scheme.get("label", scheme_name),
                "CandidateFeatureCount": len(iv_features),
                "SelectedFeatureCount": len(selected),
                "WOEFeatureCount": len(woe_data["features"]) if woe_data is not None else 0,
            }
        )

        for model_name, model_cfg in enabled_models(config).items():
            matrices = woe_data if model_name == "woe_lr" else tree
            if matrices is None:
                continue
            params = model_cfg.get("params", {})
            if model_name == "xgboost" and model_cfg.get("auto_scale_pos_weight", False):
                pos_count = float(matrices["y_train"].sum())
                neg_count = float(len(matrices["y_train"]) - pos_count)
                if pos_count > 0:
                    params = dict(params)
                    params["scale_pos_weight"] = neg_count / pos_count
            model = model_mod.make_model(model_name, params, config["project"].get("random_state", 42))
            logger.info("Scheme %s model %s training started.", scheme_name, model_name)
            if model_name == "xgboost":
                model_mod.fit_xgboost(
                    model,
                    matrices["X_train"],
                    matrices["y_train"],
                    matrices["X_valid"],
                    matrices["y_valid"],
                    model_cfg.get("early_stopping_rounds"),
                )
            else:
                model.fit(matrices["X_train"], matrices["y_train"])
            logger.info("Scheme %s model %s training finished.", scheme_name, model_name)
            model_mod.save_model(model, scheme_dir / "models" / f"{model_name}.joblib")
            logger.info("Scheme %s model %s evaluating.", scheme_name, model_name)
            valid_score = predict_score(model, matrices["X_valid"])
            threshold = find_best_threshold(matrices["y_valid"], valid_score)
            metric_rows, topk, preds = evaluate_model(
                model,
                scheme_name,
                model_name,
                matrices,
                threshold,
                config["evaluation"].get("top_rates", [0.01, 0.03, 0.05, 0.10]),
            )
            all_metric_rows.extend(metric_rows)
            all_topk.append(topk)
            all_predictions.append(preds)
            pd.DataFrame(metric_rows).to_csv(
                scheme_dir / "metrics" / f"{model_name}_metrics.csv", index=False, encoding="utf-8-sig"
            )
            topk.to_csv(scheme_dir / "metrics" / f"{model_name}_topk.csv", index=False, encoding="utf-8-sig")
            preds.to_csv(scheme_dir / "predictions" / f"{model_name}_predictions.csv", index=False, encoding="utf-8-sig")
            save_json(
                {"scheme": scheme, "model": model_name, "threshold": threshold, "params": params},
                scheme_dir / "models" / f"{model_name}_run_config.json",
            )
            logger.info("Scheme %s model %s saved. threshold=%.6f", scheme_name, model_name, threshold)
        logger.info("Scheme finished: %s", scheme_name)

    pd.DataFrame(scheme_rows).to_csv(output_dir / "tables" / "experiment_scheme_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_metric_rows).to_csv(output_dir / "tables" / "all_metrics_train_valid_test.csv", index=False, encoding="utf-8-sig")
    if all_topk:
        pd.concat(all_topk, ignore_index=True).to_csv(output_dir / "tables" / "all_topk_train_valid_test_with_lift.csv", index=False, encoding="utf-8-sig")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "predictions" / "all_predictions.csv", index=False, encoding="utf-8-sig")
    logger.info("Training finished. schemes=%d, metric_rows=%d", len(scheme_rows), len(all_metric_rows))


def run_evaluation(config: dict, experiment: str, output_dir: str | Path = "outputs") -> None:
    output_dir = resolve_path(output_dir)
    metrics_path = output_dir / "tables" / "all_metrics_train_valid_test.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run scripts/train.py before evaluate.py.")
    metrics = pd.read_csv(metrics_path)
    subset = metrics[metrics["Scheme"] == experiment]
    ensure_dir(output_dir / experiment / "metrics")
    subset.to_csv(output_dir / experiment / "metrics" / "metrics_summary.csv", index=False, encoding="utf-8-sig")


def run_explain(config: dict, experiment: str, output_dir: str | Path = "outputs") -> None:
    output_dir = resolve_path(output_dir)
    train, valid, test, iv_features, force_categorical, _risk_system = prepare_base_features(config)
    target = config["project"].get("target", "isFraud")
    selected = pd.read_csv(output_dir / experiment / "tables" / "selected_features.csv")["Feature"].tolist()
    feature_info = pd.read_csv(output_dir / experiment / "tables" / "model_feature_info.csv")
    tree = feature_mod.prepare_tree_matrices(train, valid, test, selected, target, [f for f in force_categorical if f in selected])
    model_name = config["explain"].get("model", "xgboost")
    model = model_mod.load_model(output_dir / experiment / "models" / f"{model_name}.joblib")
    shap_dir = ensure_dir(output_dir / experiment / "shap")
    sample = explain_mod.sample_for_shap(
        tree["X_test"],
        config["explain"].get("sample_size", 5000),
        config["project"].get("random_state", 42),
        output_dir / "tables" / "shap_sample_indices.csv",
    )
    y_sample = tree["y_test"].loc[sample.index]
    values = explain_mod.compute_tree_shap(model, sample)
    importance, fraud_normal = explain_mod.shap_importance_table(sample, values, feature_info, y_sample)
    importance.to_csv(shap_dir / "shap_importance.csv", index=False, encoding="utf-8-sig")
    explain_mod.mechanism_summary(importance).to_csv(shap_dir / "shap_mechanism_summary.csv", index=False, encoding="utf-8-sig")
    if fraud_normal is not None:
        fraud_normal.to_csv(shap_dir / "fraud_normal_shap_compare.csv", index=False, encoding="utf-8-sig")
    explain_mod.save_bar_plot(importance, shap_dir / "shap_importance_top30.png", config["explain"].get("max_display", 30))
    explain_mod.save_summary_plot(values, sample, shap_dir / "shap_summary_top30.png", config["explain"].get("max_display", 30))

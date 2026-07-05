# -*- coding: utf-8 -*-
"""Non-interactive RF prediction, replicating uncertainty_sampling_reg+cls.py
(steps 1-3 only: train final model on full training CSV, predict on new CSV).
No plotting, no MP API.

Usage:
  python round2_predict.py <task> <json> <train_csv> <new_csv> <out_csv>
    task      : 'classification' or 'regression'
    json      : feature_importance_results_*.json (selected_features, best_params, thr)
    train_csv : labeled training CSV (ID + features + reference target)
    new_csv   : featurized prediction targets (ID + features + reference placeholder)
    out_csv   : output path
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def main():
    task = sys.argv[1]
    json_file = sys.argv[2]
    train_file = sys.argv[3]
    new_data_file = sys.argv[4]
    output_file = sys.argv[5]

    with open(json_file, "r") as f:
        results = json.load(f)

    # JSON may carry its own task_type; CLI arg takes precedence
    task_type = task if task else results.get("task_type", "regression")

    best_indices = results["selected_features"]
    best_params = results["best_params"]
    optimal_threshold = results.get("performance_metrics", {}).get(
        "optimal_threshold", 0.5
    )
    print(f"task={task_type} | selected_features={len(best_indices)} | "
          f"threshold={optimal_threshold}")

    # --- train final model on full training set ---
    df = pd.read_csv(train_file)
    df_clean = df.dropna()
    X_full = df_clean.iloc[:, 1:-1]
    y_full = df_clean.iloc[:, -1]
    if task_type == "classification":
        y_full = (y_full >= 0).astype(int)
    print(f"train samples={len(X_full)} | features={X_full.shape[1]}")

    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    X_selected = X_full_scaled[:, best_indices]

    if task_type == "regression":
        rf_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=4)
    else:
        rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=4)
    rf_model.fit(X_selected, y_full)

    # --- predict on new data ---
    df_new = pd.read_csv(new_data_file)
    X_new = df_new.iloc[:, 1:-1]
    sample_ids = df_new.iloc[:, 0]
    X_new_scaled = scaler.transform(X_new)
    X_new_selected = X_new_scaled[:, best_indices]
    print(f"new samples={len(X_new)}")

    if task_type == "regression":
        tree_predictions = np.array(
            [tree.predict(X_new_selected) for tree in rf_model.estimators_]
        )
        predictions_mean = tree_predictions.mean(axis=0)
        predictions_std = tree_predictions.std(axis=0)
        out = pd.DataFrame({
            "ID": sample_ids,
            "prediction_log10": predictions_mean,
            "prediction_original": 10 ** predictions_mean,
            "uncertainty": predictions_std,
        })
    else:
        tree_proba = np.array(
            [tree.predict_proba(X_new_selected) for tree in rf_model.estimators_]
        )  # (n_trees, n_samples, 2)
        proba_mean = tree_proba.mean(axis=0)
        predictions_mean = proba_mean[:, 1]
        predictions_class = (predictions_mean >= optimal_threshold).astype(int)
        predictions_std = tree_proba[:, :, 1].std(axis=0)
        out = pd.DataFrame({
            "ID": sample_ids,
            "prediction_class": predictions_class,
            "prediction_probability": predictions_mean,
            "uncertainty": predictions_std,
        })
        print(f"easy-axis (class=1): {int(predictions_class.sum())}/{len(predictions_class)}")

    out.to_csv(output_file, index=False)
    print(f"saved -> {output_file}")


if __name__ == "__main__":
    main()

from __future__ import print_function

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import (
    DEPTH_ZONES,
    FEATURE_COLUMNS,
    clean_ml_rows,
    evaluate_predictions,
    fit_ridge_model,
    group_kfold_splits,
    load_config,
    predict_ridge,
    project_path,
    read_csv,
    training_ranges,
    write_csv,
    write_json,
    write_pickle,
)


def zones_from_config(config):
    zones = {}
    for key, value in config.get("zones", DEPTH_ZONES).items():
        zones[key] = (float(value[0]), float(value[1]))
    return zones


def make_sklearn_model(config):
    model_config = config.get("model", {})
    params = model_config.get("xgboost_params", {})
    try:
        from xgboost import XGBRegressor

        return "xgboost", XGBRegressor(**params)
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return "hist_gradient_boosting", HistGradientBoostingRegressor(
            max_iter=int(params.get("n_estimators", 600)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=int(config.get("random_state", 42)),
        )
    except Exception:
        return None, None


def fit_predict_sklearn(model, train_rows, test_rows, target_column):
    import numpy as np
    from sklearn.base import clone

    X_train = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in train_rows], dtype=float)
    y_train = np.asarray([float(row[target_column]) for row in train_rows], dtype=float)
    X_test = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in test_rows], dtype=float)
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    return fitted, fitted.predict(X_test).tolist()


def fit_final_sklearn(model, rows, target_column):
    import numpy as np
    from sklearn.base import clone

    X = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([float(row[target_column]) for row in rows], dtype=float)
    fitted = clone(model)
    fitted.fit(X, y)
    return fitted


def save_model(path, model):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "wb") as handle:
        pickle.dump(model, handle, protocol=2)


def train_with_backend(rows, config):
    backend, sklearn_model = make_sklearn_model(config)
    group_column = config.get("group_column", "scenario_id")
    splits = group_kfold_splits(rows, group_column, int(config.get("n_splits", 5)), int(config.get("random_state", 42)))
    cv_rows = []
    if backend:
        for fold_idx, (train_rows, test_rows) in enumerate(splits, start=1):
            _, pred_qb = fit_predict_sklearn(sklearn_model, train_rows, test_rows, "target_residual_qb_MPa")
            _, pred_qs = fit_predict_sklearn(sklearn_model, train_rows, test_rows, "target_residual_qs_kPa")
            for i, row in enumerate(test_rows):
                out = row.copy()
                out["fold"] = str(fold_idx)
                out["pred_residual_qb_MPa"] = "%.12g" % pred_qb[i]
                out["pred_residual_qs_kPa"] = "%.12g" % pred_qs[i]
                cv_rows.append(out)
        qb_model = fit_final_sklearn(sklearn_model, rows, "target_residual_qb_MPa")
        qs_model = fit_final_sklearn(sklearn_model, rows, "target_residual_qs_kPa")
        return backend, qb_model, qs_model, cv_rows

    l2 = float(config.get("model", {}).get("ridge_l2", 1.0e-6))
    for fold_idx, (train_rows, test_rows) in enumerate(splits, start=1):
        qb_model = fit_ridge_model(train_rows, FEATURE_COLUMNS, "target_residual_qb_MPa", l2)
        qs_model = fit_ridge_model(train_rows, FEATURE_COLUMNS, "target_residual_qs_kPa", l2)
        for row in test_rows:
            out = row.copy()
            out["fold"] = str(fold_idx)
            out["pred_residual_qb_MPa"] = "%.12g" % predict_ridge(qb_model, row)
            out["pred_residual_qs_kPa"] = "%.12g" % predict_ridge(qs_model, row)
            cv_rows.append(out)
    qb_model = fit_ridge_model(rows, FEATURE_COLUMNS, "target_residual_qb_MPa", l2)
    qs_model = fit_ridge_model(rows, FEATURE_COLUMNS, "target_residual_qs_kPa", l2)
    return "standard_library_ridge", qb_model, qs_model, cv_rows


def main():
    parser = argparse.ArgumentParser(description="Train residual correction models.")
    parser.add_argument("--config", default=project_path("configs", "training.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = project_path(config.get("dataset_csv", "data/processed/paired_dataset.csv"))
    rows = clean_ml_rows(read_csv(dataset_path))
    if not rows:
        raise SystemExit("No clean ML rows found in %s" % dataset_path)

    backend, qb_model, qs_model, cv_rows = train_with_backend(rows, config)
    metrics = evaluate_predictions(cv_rows, zones_from_config(config))

    save_model(project_path(config.get("qb_model", "models/qb_model.pkl")), qb_model)
    save_model(project_path(config.get("qs_model", "models/qs_model.pkl")), qs_model)
    write_json(project_path(config.get("feature_columns_json", "models/feature_columns.json")), FEATURE_COLUMNS)
    write_json(
        project_path(config.get("training_report_json", "models/training_report.json")),
        {
            "backend": backend,
            "n_rows": len(rows),
            "n_scenarios": len(set(row.get("scenario_id", "") for row in rows)),
            "feature_columns": FEATURE_COLUMNS,
            "training_ranges": training_ranges(rows),
        },
    )
    write_csv(project_path(config.get("cv_predictions_csv", "data/processed/cv_predictions.csv")), cv_rows)
    write_csv(project_path(config.get("metrics_csv", "reports/tables/metrics.csv")), metrics)
    print("Trained qb/qs models with backend=%s on %d rows." % (backend, len(rows)))


if __name__ == "__main__":
    main()

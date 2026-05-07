from __future__ import print_function

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import DEPTH_ZONES, FEATURE_COLUMNS, clean_ml_rows, evaluate_predictions, load_config, predict_ridge, project_path, read_csv, read_json, write_csv


def load_model(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_model(model, row):
    if isinstance(model, dict) and model.get("model_type") == "standard_library_ridge":
        return predict_ridge(model, row)
    import numpy as np

    X = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS]], dtype=float)
    return float(model.predict(X)[0])


def zones_from_config(config):
    zones = {}
    for key, value in config.get("zones", DEPTH_ZONES).items():
        zones[key] = (float(value[0]), float(value[1]))
    return zones


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained correction models on the paired dataset.")
    parser.add_argument("--config", default=project_path("configs", "training.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    rows = clean_ml_rows(read_csv(project_path(config.get("dataset_csv", "data/processed/paired_dataset.csv"))))
    qb_model = load_model(project_path(config.get("qb_model", "models/qb_model.pkl")))
    qs_model = load_model(project_path(config.get("qs_model", "models/qs_model.pkl")))

    predictions = []
    for row in rows:
        out = row.copy()
        out["pred_residual_qb_MPa"] = "%.12g" % predict_model(qb_model, row)
        out["pred_residual_qs_kPa"] = "%.12g" % predict_model(qs_model, row)
        predictions.append(out)
    metrics = evaluate_predictions(predictions, zones_from_config(config))
    write_csv(project_path(config.get("evaluation_predictions_csv", "data/processed/evaluation_predictions.csv")), predictions)
    write_csv(project_path(config.get("metrics_csv", "reports/tables/metrics.csv")), metrics)
    print("Evaluated %d rows." % len(predictions))


if __name__ == "__main__":
    main()

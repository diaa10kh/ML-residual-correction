from __future__ import print_function

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import (
    FEATURE_COLUMNS,
    add_curve_features,
    classify_validity,
    extrapolation_warnings,
    load_config,
    predict_ridge,
    project_path,
    read_csv,
    read_json,
    safe_float,
    write_csv,
)


def load_model(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict_model(model, row):
    if isinstance(model, dict) and model.get("model_type") == "standard_library_ridge":
        return predict_ridge(model, row)
    import numpy as np

    X = np.asarray([[float(row[col]) for col in FEATURE_COLUMNS]], dtype=float)
    return float(model.predict(X)[0])


def main():
    parser = argparse.ArgumentParser(description="Apply trained residual correction models to one resampled high-S run.")
    parser.add_argument("--config", default=project_path("configs", "training.yaml"))
    parser.add_argument("--input", required=True, help="Resampled high-S run CSV.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    report = read_json(project_path(config.get("training_report_json", "models/training_report.json")))
    qb_model = load_model(project_path(config.get("qb_model", "models/qb_model.pkl")))
    qs_model = load_model(project_path(config.get("qs_model", "models/qs_model.pkl")))
    rows = add_curve_features(read_csv(args.input))

    corrected = []
    for row in rows:
        out = row.copy()
        pred_qb = predict_model(qb_model, row)
        pred_qs = predict_model(qs_model, row)
        out["pred_residual_qb_MPa"] = "%.12g" % pred_qb
        out["pred_residual_qs_kPa"] = "%.12g" % pred_qs
        out["qb_corrected_MPa"] = "%.12g" % (safe_float(row.get("qb_MPa")) + pred_qb)
        out["qs_corrected_kPa"] = "%.12g" % (safe_float(row.get("qs_kPa")) + pred_qs)
        corrected.append(out)

    output = args.output
    if not output:
        run_id = corrected[0].get("run_id", "corrected") if corrected else "corrected"
        output = project_path("data", "processed", "corrected_%s.csv" % run_id)
    write_csv(output, corrected)

    max_energy = max([safe_float(row.get("energy_ratio_cummax"), safe_float(row.get("energy_ratio"))) for row in corrected])
    warnings = extrapolation_warnings(corrected, report.get("training_ranges", {}))
    print("Wrote corrected curve: %s" % output)
    print("Validity gate: %s (max ALLKE/ALLIE=%.6g)" % (classify_validity(max_energy), max_energy))
    for warning in warnings:
        print("WARNING: correction is extrapolative: %s" % warning)


if __name__ == "__main__":
    main()

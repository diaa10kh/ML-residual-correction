from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import load_config, project_path, read_csv, safe_float


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise SystemExit("matplotlib is required for plots: %s" % exc)


def group_by(rows, key):
    out = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def line_plot(plt, rows, output, title, y_columns, ylabel):
    plt.figure(figsize=(7.0, 5.0))
    rows = sorted(rows, key=lambda row: safe_float(row.get("eta")))
    eta = [safe_float(row.get("eta")) for row in rows]
    for label, column in y_columns:
        values = [safe_float(row.get(column)) for row in rows]
        plt.plot(values, eta, label=label)
    plt.gca().invert_yaxis()
    plt.xlabel(ylabel)
    plt.ylabel("eta = z/L")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    folder = os.path.dirname(output)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def benchmark_plots(plt, predictions):
    scenario = "G0_DENS_HIGH_V_REF"
    rows = [row for row in predictions if row.get("scenario_id") == scenario and row.get("S") in ["50", "100"]]
    if not rows:
        return []
    out_paths = []
    by_run = group_by(rows, "run_id")
    merged = []
    for run_id in sorted(by_run.keys()):
        for row in by_run[run_id]:
            prefix = "S%s" % row.get("S")
            item = row.copy()
            item["label_prefix"] = prefix
            item["qb_corrected_MPa"] = safe_float(row.get("qb_fast_MPa")) + safe_float(row.get("pred_residual_qb_MPa"))
            item["qs_corrected_kPa"] = safe_float(row.get("qs_fast_kPa")) + safe_float(row.get("pred_residual_qs_kPa"))
            merged.append(item)
    for quantity, unit in [("qb", "MPa"), ("qs", "kPa")]:
        plt.figure(figsize=(7.0, 5.0))
        ref_done = False
        for run_id in sorted(by_run.keys()):
            run_rows = sorted([row for row in merged if row.get("run_id") == run_id], key=lambda row: safe_float(row.get("eta")))
            eta = [safe_float(row.get("eta")) for row in run_rows]
            s_value = run_rows[0].get("S")
            if not ref_done:
                plt.plot([safe_float(row.get("%s_S1_%s" % (quantity, unit))) for row in run_rows], eta, label="S=1 reference")
                ref_done = True
            plt.plot([safe_float(row.get("%s_fast_%s" % (quantity, unit))) for row in run_rows], eta, "--", label="S=%s uncorrected" % s_value)
            plt.plot([safe_float(row.get("%s_corrected_%s" % (quantity, unit))) for row in run_rows], eta, label="S=%s corrected" % s_value)
        plt.gca().invert_yaxis()
        plt.xlabel("%s [%s]" % (quantity, unit))
        plt.ylabel("eta = z/L")
        plt.title("%s benchmark correction" % scenario)
        plt.grid(True, alpha=0.3)
        plt.legend()
        output = project_path("reports", "figures", "benchmark_%s.png" % quantity)
        plt.tight_layout()
        plt.savefig(output, dpi=220)
        plt.close()
        out_paths.append(output)
    return out_paths


def improvement_plot(plt, metrics):
    rows = [row for row in metrics if row.get("zone") == "full"]
    if not rows:
        return None
    by_s = {}
    for row in rows:
        key = (row.get("S"), row.get("quantity"))
        by_s.setdefault(key, []).append(safe_float(row.get("improvement_ratio")))
    labels = sorted(by_s.keys())
    values = [sum(by_s[key]) / max(len(by_s[key]), 1) for key in labels]
    x = list(range(len(labels)))
    plt.figure(figsize=(8.0, 4.5))
    plt.bar(x, values)
    plt.xticks(x, ["S%s %s" % (key[0], key[1]) for key in labels], rotation=45, ha="right")
    plt.ylabel("uncorrected RMSE / corrected RMSE")
    plt.title("Correction improvement by S")
    plt.tight_layout()
    output = project_path("reports", "figures", "improvement_by_S.png")
    plt.savefig(output, dpi=220)
    plt.close()
    return output


def main():
    parser = argparse.ArgumentParser(description="Create report figures from evaluation outputs.")
    parser.add_argument("--config", default=project_path("configs", "training.yaml"))
    args = parser.parse_args()

    plt = require_matplotlib()
    config = load_config(args.config)
    prediction_path = project_path(config.get("evaluation_predictions_csv", "data/processed/evaluation_predictions.csv"))
    metric_path = project_path(config.get("metrics_csv", "reports/tables/metrics.csv"))
    outputs = []
    if os.path.exists(prediction_path):
        outputs.extend(benchmark_plots(plt, read_csv(prediction_path)))
    if os.path.exists(metric_path):
        path = improvement_plot(plt, read_csv(metric_path))
        if path:
            outputs.append(path)
    print("Wrote %d figure(s)." % len(outputs))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

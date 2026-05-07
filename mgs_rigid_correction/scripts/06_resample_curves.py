from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import eta_grid_from_config, load_config, project_path, read_csv, resample_run, write_csv


def main():
    parser = argparse.ArgumentParser(description="Resample per-run extracted curves to a common eta grid.")
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata.csv"))
    parser.add_argument("--matrix-config", default=project_path("configs", "matrix_phase1.yaml"))
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    eta_grid = eta_grid_from_config(load_config(args.matrix_config))
    metadata = read_csv(args.metadata)
    completed = 0
    skipped = 0
    for row in metadata:
        if args.run_id and row.get("run_id") != args.run_id:
            continue
        source = row.get("postprocess_csv", "")
        if not os.path.exists(source):
            skipped += 1
            continue
        run_rows = read_csv(source)
        output = row.get("resampled_csv") or project_path("data", "processed", "resampled_runs", row["run_id"] + ".csv")
        resampled = resample_run(run_rows, row, eta_grid)
        write_csv(output, resampled)
        completed += 1
    print("Resampled %d run(s); skipped %d missing extracted CSV(s)." % (completed, skipped))


if __name__ == "__main__":
    main()

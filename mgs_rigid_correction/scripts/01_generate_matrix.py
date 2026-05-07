from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import ensure_project_dirs, expand_matrix, load_config, project_path, write_csv


def main():
    parser = argparse.ArgumentParser(description="Generate rigid-pile MGS run metadata.")
    parser.add_argument("--config", default=project_path("configs", "matrix_phase1.yaml"))
    parser.add_argument("--output", default=project_path("data", "extracted", "run_metadata.csv"))
    parser.add_argument(
        "--soil-model",
        choices=["Hypoplastisch", "Mohr-Coulomb"],
        default="",
        help="Override the soil_model in the matrix config.",
    )
    args = parser.parse_args()

    ensure_project_dirs()
    config = load_config(args.config)
    if args.soil_model:
        config["soil_model"] = args.soil_model
    rows = expand_matrix(config)
    write_csv(args.output, rows)

    scenarios = sorted(set(row["scenario_id"] for row in rows))
    print("Wrote %d runs across %d scenarios: %s" % (len(rows), len(scenarios), args.output))


if __name__ == "__main__":
    main()

from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import build_paired_dataset, clean_ml_rows, project_path, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description="Build paired high-MGS/S=1 residual ML dataset.")
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata.csv"))
    parser.add_argument("--output", default=project_path("data", "processed", "paired_dataset.csv"))
    args = parser.parse_args()

    metadata = read_csv(args.metadata)
    rows = []
    missing = 0
    for row in metadata:
        path = row.get("resampled_csv", "")
        if not os.path.exists(path):
            missing += 1
            continue
        rows.extend(read_csv(path))
    if not rows:
        raise SystemExit("No resampled runs found")
    dataset = clean_ml_rows(build_paired_dataset(rows))
    write_csv(args.output, dataset)
    print("Wrote %d ML rows to %s; missing resampled runs=%d" % (len(dataset), args.output, missing))


if __name__ == "__main__":
    main()

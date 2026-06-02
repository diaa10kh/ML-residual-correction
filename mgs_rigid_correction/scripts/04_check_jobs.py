from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import postprocess_csv_path, project_path, read_csv, resolve_project_path, write_csv


def exists(path):
    return bool(path) and os.path.exists(path)


def main():
    parser = argparse.ArgumentParser(description="Check expected Abaqus files for each run.")
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata_full.csv"))
    parser.add_argument("--output", default=project_path("reports", "tables", "job_status.csv"))
    args = parser.parse_args()

    metadata_path = resolve_project_path(args.metadata)
    output_path = resolve_project_path(args.output)
    rows = read_csv(metadata_path)
    status_rows = []
    counts = {"inp": 0, "odb": 0, "csv": 0}
    for row in rows:
        csv_path = postprocess_csv_path(row)
        status = {
            "run_id": row.get("run_id", ""),
            "scenario_id": row.get("scenario_id", ""),
            "S": row.get("S", ""),
            "input_file": row.get("input_file", ""),
            "odb_file": row.get("odb_file", ""),
            "postprocess_csv": csv_path,
            "inp_exists": str(exists(row.get("input_file", ""))),
            "odb_exists": str(exists(row.get("odb_file", ""))),
            "csv_exists": str(exists(csv_path)),
        }
        if status["inp_exists"] == "True":
            counts["inp"] += 1
        if status["odb_exists"] == "True":
            counts["odb"] += 1
        if status["csv_exists"] == "True":
            counts["csv"] += 1
        status_rows.append(status)
    write_csv(output_path, status_rows)
    print("Checked %d runs. inp=%d odb=%d extracted_csv=%d" % (len(rows), counts["inp"], counts["odb"], counts["csv"]))
    print("Wrote %s" % output_path)


if __name__ == "__main__":
    main()

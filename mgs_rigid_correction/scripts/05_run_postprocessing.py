from __future__ import print_function

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import mkdir_p, project_path, read_csv, row_to_jsonable, write_json


def selected_rows(rows, run_id, limit, require_odb):
    out = []
    for row in rows:
        if run_id and row.get("run_id") != run_id:
            continue
        if require_odb and not os.path.exists(row.get("odb_file", "")):
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def run_abaqus(abaqus_cmd, postprocessor_script, config_path):
    if os.name == "nt":
        command = '"%s" cae noGUI="%s" -- --config "%s"' % (abaqus_cmd, postprocessor_script, config_path)
        return subprocess.call(command, shell=True)
    cmd = [abaqus_cmd, "cae", "noGUI=%s" % postprocessor_script, "--", "--config", config_path]
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Prepare and optionally run Abaqus postprocessing.")
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata.csv"))
    parser.add_argument("--postprocessor-script", default=os.path.abspath(os.path.join(project_path(), os.pardir, "04_postprocessing.py")))
    parser.add_argument("--abaqus-cmd", default="abaqus")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--odb-path", default="", help="Optional one-run ODB override, useful for testing upgraded ODB copies.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--require-odb", action="store_true")
    args = parser.parse_args()

    rows = selected_rows(read_csv(args.metadata), args.run_id, args.limit, args.require_odb)
    if not rows:
        raise SystemExit("No rows selected for postprocessing")
    if args.odb_path and len(rows) != 1:
        raise SystemExit("--odb-path can only be used when exactly one row is selected")

    prepared = 0
    for row in rows:
        run_id = row["run_id"]
        run_dir = row["run_dir"]
        mkdir_p(run_dir)
        config_path = os.path.join(run_dir, "postprocess_config.json")
        config = row_to_jsonable(row)
        config["output_csv"] = row.get("postprocess_csv")
        config["odb_path"] = os.path.abspath(args.odb_path) if args.odb_path else row.get("odb_file")
        config["sta_path"] = row.get("sta_file")
        config["force_unit_scale_to_kN"] = 1.0
        write_json(config_path, config)
        if args.execute:
            if config.get("output_csv") and os.path.exists(config["output_csv"]):
                os.remove(config["output_csv"])
            rc = run_abaqus(args.abaqus_cmd, os.path.abspath(args.postprocessor_script), config_path)
            if rc != 0:
                raise SystemExit("Abaqus postprocessing failed for %s with exit code %s" % (run_id, rc))
            if config.get("output_csv") and not os.path.exists(config["output_csv"]):
                raise SystemExit("Abaqus returned without creating expected CSV: %s" % config["output_csv"])
        prepared += 1

    action = "Postprocessed" if args.execute else "Prepared postprocess configs for"
    print("%s %d run(s)." % (action, prepared))


if __name__ == "__main__":
    main()

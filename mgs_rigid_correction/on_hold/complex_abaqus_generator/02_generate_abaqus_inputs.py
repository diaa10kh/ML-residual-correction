from __future__ import print_function

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_SCRIPT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir, "scripts"))
if ACTIVE_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, ACTIVE_SCRIPT_DIR)

from mgs_common import mkdir_p, project_path, read_csv, row_to_jsonable, write_json


def selected_rows(rows, run_id, limit):
    out = []
    for row in rows:
        if run_id and row.get("run_id") != run_id:
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def write_launcher(run_dir, run_id, abaqus_cmd, generator_script, case_config):
    launcher = os.path.join(run_dir, "generate_input.ps1")
    with open(launcher, "w") as handle:
        handle.write("# Generated helper for %s\n" % run_id)
        handle.write('$env:MGS_CASE_CONFIG="%s"\n' % case_config)
        handle.write('%s cae noGUI="%s"\n' % (abaqus_cmd, generator_script))
        handle.write('Remove-Item Env:\\MGS_CASE_CONFIG -ErrorAction SilentlyContinue\n')
    return launcher


def run_abaqus(abaqus_cmd, generator_script, case_config):
    env = os.environ.copy()
    env["MGS_CASE_CONFIG"] = case_config
    log_path = os.path.join(os.path.dirname(case_config), "abaqus_input_generation.log")
    log_handle = open(log_path, "w")
    if os.name == "nt":
        command = '"%s" cae noGUI="%s"' % (abaqus_cmd, generator_script)
        try:
            return subprocess.call(command, shell=True, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        finally:
            log_handle.close()
    cmd = [abaqus_cmd, "cae", "noGUI=%s" % generator_script]
    try:
        return subprocess.call(cmd, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
    finally:
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Create Abaqus case JSON files and optionally write .inp files.")
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata_phase0.csv"))
    parser.add_argument(
        "--generator-script",
        default=project_path("on_hold", "complex_abaqus_generator", "00_3D_CPT_deek.py"),
    )
    parser.add_argument("--abaqus-cmd", default="abaqus")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--soil-model",
        choices=["Hypoplastisch", "Mohr-Coulomb"],
        default="",
        help="Override soil_model when writing case_config.json files.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually call Abaqus/CAE to write .inp files.")
    parser.add_argument("--overwrite-config", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.metadata)
    rows = selected_rows(rows, args.run_id, args.limit)
    if not rows:
        raise SystemExit("No rows selected from %s" % args.metadata)

    generated = 0
    for row in rows:
        run_id = row["run_id"]
        run_dir = row["run_dir"]
        mkdir_p(run_dir)
        case_config = row.get("case_config_file") or os.path.join(run_dir, "case_config.json")
        if args.overwrite_config or not os.path.exists(case_config):
            config = row_to_jsonable(row)
            if args.soil_model:
                config["soil_model"] = args.soil_model
            config["workspace_root"] = os.path.abspath(os.path.join(project_path(), os.pardir))
            config["run_dir"] = run_dir
            config["num_cpus"] = int(float(config.get("num_cpus") or 8))
            config["enable_energy_output"] = True
            config["field_output_points"] = int(float(config.get("field_output_points") or 100))
            config["history_output_points"] = int(float(config.get("history_output_points") or 1000))
            write_json(case_config, config)
        write_launcher(run_dir, run_id, args.abaqus_cmd, os.path.abspath(args.generator_script), case_config)
        if args.execute:
            expected_input = row.get("input_file", "")
            if expected_input and os.path.exists(expected_input):
                os.remove(expected_input)
            rc = run_abaqus(args.abaqus_cmd, os.path.abspath(args.generator_script), case_config)
            if rc != 0:
                raise SystemExit("Abaqus input generation failed for %s with exit code %s" % (run_id, rc))
            if not os.path.exists(expected_input):
                raise SystemExit("Abaqus returned without creating expected input file: %s" % expected_input)
        generated += 1

    action = "Generated .inp files for" if args.execute else "Prepared case configs for"
    print("%s %d run(s)." % (action, generated))


if __name__ == "__main__":
    main()

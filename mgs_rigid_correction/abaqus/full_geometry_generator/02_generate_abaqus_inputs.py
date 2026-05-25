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


def normalise_soil_model(value):
    value = (value or "").strip()
    aliases = {
        "": "",
        "both": "both",
        "all": "both",
        "Hypoplastisch": "hypoplastic",
        "hypoplastic": "hypoplastic",
        "Mohr-Coulomb": "mcm",
        "mohr_coulomb": "mcm",
        "mcm": "mcm",
    }
    return aliases.get(value, value)


def abaqus_soil_label(value):
    key = normalise_soil_model(value)
    if key == "hypoplastic":
        return "Hypoplastisch"
    if key == "mcm":
        return "Mohr-Coulomb"
    return ""


def selected_rows(rows, run_id, limit, soil_model, one_per_geometry, density_id, velocity_id, s_value):
    out = []
    for row in rows:
        if run_id and row.get("run_id") != run_id:
            continue
        if soil_model not in ("", "both"):
            row_soil_model = normalise_soil_model(row.get("soil_model", ""))
            if row_soil_model != soil_model:
                continue
        if one_per_geometry:
            if row.get("density_id", "") != density_id:
                continue
            if row.get("velocity_id", "") != velocity_id:
                continue
            if int(float(row.get("S", 0))) != int(s_value):
                continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    if not one_per_geometry:
        return out

    by_geometry = {}
    for row in out:
        geometry_id = row.get("geometry_id", "")
        if geometry_id and geometry_id not in by_geometry:
            by_geometry[geometry_id] = row
    return [by_geometry[key] for key in sorted(by_geometry.keys())]


def truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def cae_path_for_run(run_dir, run_id):
    return os.path.join(run_dir, run_id + ".cae")


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
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata_full.csv"))
    parser.add_argument(
        "--generator-script",
        default=project_path("abaqus", "full_geometry_generator", "00_3D_CPT_deek.py"),
    )
    parser.add_argument("--abaqus-cmd", default="abaqus")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--one-per-geometry",
        action="store_true",
        help="Select one representative run per geometry for geometry smoke testing.",
    )
    parser.add_argument("--representative-density", default="DENS_REF")
    parser.add_argument("--representative-velocity", default="V_REF")
    parser.add_argument("--representative-s", type=int, default=1)
    parser.add_argument(
        "--save-cae",
        choices=["none", "first", "all"],
        default="none",
        help="Save no CAE file, only the first selected CAE file, or all selected CAE files.",
    )
    parser.add_argument(
        "--soil-model",
        choices=["hypoplastic", "mcm", "mohr_coulomb", "Hypoplastisch", "Mohr-Coulomb", "both", "all", ""],
        default="",
        help="Filter/override soil_model when writing case_config.json files. Use both/all to keep metadata values.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually call Abaqus/CAE to write .inp files.")
    parser.add_argument("--overwrite-config", action="store_true")
    args = parser.parse_args()

    selected_soil_model = normalise_soil_model(args.soil_model)
    rows = read_csv(args.metadata)
    rows = selected_rows(
        rows,
        args.run_id,
        args.limit,
        selected_soil_model,
        args.one_per_geometry,
        args.representative_density,
        args.representative_velocity,
        args.representative_s,
    )
    if not rows:
        raise SystemExit("No rows selected from %s" % args.metadata)

    generated = 0
    for row in rows:
        run_id = row["run_id"]
        run_dir = row["run_dir"]
        mkdir_p(run_dir)
        case_config = row.get("case_config_file") or os.path.join(run_dir, "case_config.json")
        save_cae = args.save_cae == "all" or (args.save_cae == "first" and generated == 0)
        if args.overwrite_config or not os.path.exists(case_config):
            config = row_to_jsonable(row)
            override_label = abaqus_soil_label(selected_soil_model)
            if override_label:
                config["soil_model"] = override_label
            config["workspace_root"] = os.path.abspath(os.path.join(project_path(), os.pardir))
            config["run_dir"] = run_dir
            config["num_cpus"] = int(float(config.get("num_cpus") or 8))
            config["enable_energy_output"] = True
            config["field_output_points"] = int(float(config.get("field_output_points") or 100))
            config["history_output_points"] = int(float(config.get("history_output_points") or 1000))
            config["save_cae"] = bool(save_cae)
            config["cae_file"] = cae_path_for_run(run_dir, run_id) if save_cae else ""
            write_json(case_config, config)
        write_launcher(run_dir, run_id, args.abaqus_cmd, os.path.abspath(args.generator_script), case_config)
        if args.execute:
            expected_input = row.get("input_file", "")
            if expected_input and os.path.exists(expected_input):
                os.remove(expected_input)
            expected_cae = cae_path_for_run(run_dir, run_id) if save_cae else ""
            if expected_cae and os.path.exists(expected_cae):
                os.remove(expected_cae)
            rc = run_abaqus(args.abaqus_cmd, os.path.abspath(args.generator_script), case_config)
            if rc != 0:
                raise SystemExit("Abaqus input generation failed for %s with exit code %s" % (run_id, rc))
            if not os.path.exists(expected_input):
                raise SystemExit("Abaqus returned without creating expected input file: %s" % expected_input)
            if expected_cae and not os.path.exists(expected_cae):
                raise SystemExit("Abaqus returned without creating expected CAE file: %s" % expected_cae)
        generated += 1

    action = "Generated .inp files for" if args.execute else "Prepared case configs for"
    print("%s %d run(s)." % (action, generated))
    if args.one_per_geometry:
        print(
            "Representative selection: density=%s velocity=%s S=%s"
            % (args.representative_density, args.representative_velocity, args.representative_s)
        )
    if args.save_cae != "none":
        print("CAE save mode: %s" % args.save_cae)


if __name__ == "__main__":
    main()

from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import ensure_project_dirs, expand_matrix, load_config, project_path, resolve_project_path, write_csv


FULL_CONFIGS = {
    "hypoplastic": project_path("configs", "matrix_full.yaml"),
    "mcm": project_path("configs", "matrix_full_mohr_coulomb.yaml"),
}

FULL_OUTPUTS = {
    "hypoplastic": project_path("data", "extracted", "run_metadata_full.csv"),
    "mcm": project_path("data", "extracted", "run_metadata_full_mohr_coulomb.csv"),
}


def normalise_soil_model(value):
    value = (value or "config").strip()
    aliases = {
        "config": "config",
        "Hypoplastisch": "hypoplastic",
        "hypoplastic": "hypoplastic",
        "Mohr-Coulomb": "mcm",
        "mohr_coulomb": "mcm",
        "mcm": "mcm",
        "both": "both",
        "all": "both",
    }
    return aliases.get(value, value)


def apply_soil_model_override(config, soil_model):
    if soil_model == "hypoplastic":
        config["soil_model"] = "Hypoplastisch"
        config.pop("run_id_prefix", None)
    elif soil_model == "mcm":
        config["soil_model"] = "Mohr-Coulomb"
        config["run_id_prefix"] = "MC"
    return config


def generate_one(soil_model, config_arg, output_arg):
    default_key = "hypoplastic" if soil_model == "config" else soil_model
    config_path = resolve_project_path(config_arg or FULL_CONFIGS[default_key])
    output_path = resolve_project_path(output_arg or FULL_OUTPUTS[default_key])
    config = load_config(config_path)
    if soil_model != "config":
        config = apply_soil_model_override(config, soil_model)
    rows = expand_matrix(config)
    write_csv(output_path, rows)

    label = normalise_soil_model(config.get("soil_model", soil_model))
    scenarios = sorted(set(row["scenario_id"] for row in rows))
    print("Wrote %d %s runs across %d scenarios: %s" % (len(rows), label, len(scenarios), output_path))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate full-version deformable-pile MGS run metadata.")
    parser.add_argument("--config", default="", help="Matrix config. Defaults to the selected full-version config.")
    parser.add_argument("--output", default="", help="Metadata CSV. Defaults to the selected full-version output path.")
    parser.add_argument(
        "--soil-model",
        choices=["config", "hypoplastic", "mcm", "mohr_coulomb", "Hypoplastisch", "Mohr-Coulomb", "both", "all"],
        default="config",
        help=(
            "Generate hypoplastic, MCM, or both full-version metadata files. "
            "Default respects --config; without --config it uses the full hypoplastic config."
        ),
    )
    args = parser.parse_args()

    ensure_project_dirs()
    selected = normalise_soil_model(args.soil_model)
    if selected == "both":
        if args.config or args.output:
            raise SystemExit("ERROR: --soil-model both uses the standard full hypoplastic and MCM configs/outputs.")
        generate_one("hypoplastic", "", "")
        generate_one("mcm", "", "")
        return

    generate_one(selected, args.config, args.output)


if __name__ == "__main__":
    main()

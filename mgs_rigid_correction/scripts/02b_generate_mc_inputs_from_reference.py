from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import as_float, mkdir_p, project_path, read_csv


def abaqus_number(value):
    value = float(value)
    if abs(value - round(value)) < 1.0e-12:
        return "%d." % int(round(value))
    return "%.12g" % value


def set_next_data_line(lines, keyword_index, new_line):
    index = keyword_index + 1
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("**"):
            lines[index] = new_line
            return
        index += 1
    raise RuntimeError("No data line found after %s" % lines[keyword_index].strip())


def find_line(lines, predicate, start=0, stop=None):
    if stop is None:
        stop = len(lines)
    for index in range(start, stop):
        if predicate(lines[index]):
            return index
    return -1


def find_keyword(lines, keyword, start=0, stop=None):
    key = keyword.lower()
    return find_line(lines, lambda line: line.strip().lower() == key, start, stop)


def material_block_bounds(lines, material_name):
    start = find_line(
        lines,
        lambda line: line.strip().lower() == ("*material, name=%s" % material_name).lower(),
    )
    if start < 0:
        raise RuntimeError("Material block not found: %s" % material_name)
    stop = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].strip().lower().startswith("*material, name="):
            stop = index
            break
    return start, stop


def patch_soil_material(lines, row):
    start, stop = material_block_bounds(lines, "HYPO-VW96-Sand")
    density = as_float(row["mc_density"]) * as_float(row["S"])
    elastic_E = as_float(row["mc_E"])
    elastic_nu = as_float(row["mc_nu"])
    phi = as_float(row["mc_phi"])
    psi = as_float(row["mc_psi"])
    cohesion = as_float(row["mc_cohesion"])
    plastic_strain = as_float(row.get("mc_plastic_strain"), 0.0)

    density_index = find_keyword(lines, "*Density", start, stop)
    elastic_index = find_keyword(lines, "*Elastic", start, stop)
    mc_index = find_keyword(lines, "*Mohr Coulomb", start, stop)
    hardening_index = find_keyword(lines, "*Mohr Coulomb Hardening", start, stop)
    for label, index in [
        ("Density", density_index),
        ("Elastic", elastic_index),
        ("Mohr Coulomb", mc_index),
        ("Mohr Coulomb Hardening", hardening_index),
    ]:
        if index < 0:
            raise RuntimeError("%s keyword not found in HYPO-VW96-Sand block" % label)

    set_next_data_line(lines, density_index, " %s,\n" % abaqus_number(density))
    set_next_data_line(
        lines,
        elastic_index,
        "%s, %s\n" % (abaqus_number(elastic_E), abaqus_number(elastic_nu)),
    )
    set_next_data_line(
        lines,
        mc_index,
        " %s,%s\n" % (abaqus_number(phi), abaqus_number(psi)),
    )
    set_next_data_line(
        lines,
        hardening_index,
        " %s,%s\n" % (abaqus_number(cohesion), abaqus_number(plastic_strain)),
    )


def patch_steel_material(lines, row):
    start, stop = material_block_bounds(lines, "Stahl")
    density = 7.8 * as_float(row["S"])
    density_index = find_keyword(lines, "*Density", start, stop)
    if density_index < 0:
        raise RuntimeError("Density keyword not found in Stahl block")
    set_next_data_line(lines, density_index, " %s,\n" % abaqus_number(density))


def patch_gravity(lines, row):
    gravity = 9.81 / as_float(row["S"])
    index = find_line(
        lines,
        lambda line: line.strip().lower().startswith("soil-1.soil_all, grav,"),
    )
    if index < 0:
        raise RuntimeError("Gravity Dload line not found")
    lines[index] = "Soil-1.Soil_All, GRAV, %s, 0., 0., -1.\n" % abaqus_number(gravity)


def patch_velocity_timing(lines, row):
    penetration = as_float(row["penetration_m"])
    velocity = as_float(row["velocity_m_per_s"])
    step_time = penetration / max(velocity, 1.0e-12)

    amp_index = find_keyword(lines, "*Amplitude, name=Amp_Einpressen")
    if amp_index < 0:
        raise RuntimeError("Amp_Einpressen not found")
    set_next_data_line(
        lines,
        amp_index,
        "             0.,              0.,             %s,              1.\n"
        % abaqus_number(step_time),
    )

    step_index = find_keyword(lines, "*Step, name=Einpressen, nlgeom=YES")
    if step_index < 0:
        raise RuntimeError("Einpressen step not found")
    dynamic_index = find_keyword(lines, "*Dynamic, Explicit", step_index)
    if dynamic_index < 0:
        raise RuntimeError("Einpressen dynamic keyword not found")
    set_next_data_line(lines, dynamic_index, ", %s\n" % abaqus_number(step_time))


def einpressen_output_intervals(step_time):
    if abs(step_time - 9.0) < 1.0e-9:
        return 0.1, 0.01
    return step_time / 100.0, step_time / 1000.0


def patch_einpressen_output_intervals(lines, row):
    penetration = as_float(row["penetration_m"])
    velocity = as_float(row["velocity_m_per_s"])
    step_time = penetration / max(velocity, 1.0e-12)
    field_interval, history_interval = einpressen_output_intervals(step_time)

    step_index = find_keyword(lines, "*Step, name=Einpressen, nlgeom=YES")
    if step_index < 0:
        raise RuntimeError("Einpressen step not found")
    end_index = find_keyword(lines, "*End Step", step_index)
    if end_index < 0:
        raise RuntimeError("Einpressen end step not found")

    field_index = find_line(
        lines,
        lambda line: line.strip().lower().startswith("*output, field,"),
        step_index,
        end_index,
    )
    history_index = find_line(
        lines,
        lambda line: line.strip().lower().startswith("*output, history,"),
        step_index,
        end_index,
    )
    if field_index < 0:
        raise RuntimeError("Einpressen field output keyword not found")
    if history_index < 0:
        raise RuntimeError("Einpressen history output keyword not found")

    lines[field_index] = "*Output, field, time interval=%s\n" % abaqus_number(field_interval)
    lines[history_index] = "*Output, history, time interval=%s\n" % abaqus_number(history_interval)


def patch_template(template_lines, row):
    lines = list(template_lines)
    patch_soil_material(lines, row)
    patch_steel_material(lines, row)
    patch_gravity(lines, row)
    patch_velocity_timing(lines, row)
    patch_einpressen_output_intervals(lines, row)
    return lines


def verify_mc_reference(lines):
    forbidden = ["*User Material", "*Depvar", "*Initial Conditions, type=SOLUTION"]
    text = "".join(lines).lower()
    for keyword in forbidden:
        if keyword.lower() in text:
            raise RuntimeError("Reference contains forbidden Mohr-Coulomb keyword: %s" % keyword)


def remove_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def clean_run_dir(run_dir):
    if not os.path.basename(run_dir).startswith("MC_"):
        raise RuntimeError("Refusing to clean non-MC run dir: %s" % run_dir)
    if not os.path.isdir(run_dir):
        mkdir_p(run_dir)
        return 0
    removed = 0
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
            removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Create Mohr-Coulomb Phase 0 .inp files by patching a reference .inp."
    )
    parser.add_argument(
        "--metadata",
        default=project_path("data", "extracted", "run_metadata_phase0_mohr_coulomb.csv"),
    )
    parser.add_argument(
        "--reference-inp",
        default=os.path.abspath(
            os.path.join(project_path(), os.pardir, "CPT_90_MCM_einpressen_Voll_S001.inp")
        ),
    )
    parser.add_argument("--clean", action="store_true", help="Remove old files from MC run folders first.")
    parser.add_argument(
        "--clean-derived-data",
        action="store_true",
        help="Remove stale extracted and resampled MC CSVs referenced by the metadata.",
    )
    args = parser.parse_args()

    rows = read_csv(args.metadata)
    if not rows:
        raise SystemExit("No rows in metadata: %s" % args.metadata)
    with open(args.reference_inp, "r") as handle:
        template_lines = handle.readlines()
    verify_mc_reference(template_lines)

    removed_files = 0
    written = 0
    for row in rows:
        if row.get("soil_model") != "Mohr-Coulomb" or not row.get("run_id", "").startswith("MC_"):
            raise RuntimeError("Unexpected non-Mohr-Coulomb row: %s" % row.get("run_id"))
        run_dir = row["run_dir"]
        if args.clean:
            removed_files += clean_run_dir(run_dir)
        else:
            mkdir_p(run_dir)
        if args.clean_derived_data:
            remove_if_exists(row.get("postprocess_csv", ""))
            remove_if_exists(row.get("resampled_csv", ""))
        output = row["input_file"]
        patched = patch_template(template_lines, row)
        with open(output, "w") as handle:
            handle.writelines(patched)
        written += 1

    print("Reference: %s" % args.reference_inp)
    print("Removed %d old run-folder file(s)." % removed_files)
    print("Wrote %d patched Mohr-Coulomb input file(s)." % written)


if __name__ == "__main__":
    main()

from __future__ import print_function

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mgs_common import as_float, mkdir_p, project_path, read_csv, resolve_project_path


SOIL_BASE_DENSITY = 1.64
STEEL_BASE_DENSITY = 7.8
REFERENCE_D_M = 0.60
REFERENCE_L_M = 12.0
REFERENCE_PILE_INSTANCE_Z = 27.0


def abaqus_number(value):
    value = float(value)
    if abs(value - round(value)) < 1.0e-12:
        return "%d." % int(round(value))
    return "%.12g" % value


def format_node(node_id, x, y, z):
    return "%7d, %16s, %16s, %16s\n" % (
        int(node_id),
        abaqus_number(x),
        abaqus_number(y),
        abaqus_number(z),
    )


def is_keyword(line):
    return line.lstrip().startswith("*")


def split_data_line(line):
    return [part.strip() for part in line.split(",")]


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


def patch_part_nodes(lines, part_name, radius_scale, z_scale=1.0):
    in_part = False
    in_nodes = False
    part_marker = "name=%s" % part_name.lower()
    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("*part") and part_marker in lower:
            in_part = True
            in_nodes = False
            continue
        if in_part and lower.startswith("*end part"):
            in_part = False
            in_nodes = False
            continue
        if in_part and lower.startswith("*node"):
            in_nodes = True
            continue
        if in_part and in_nodes and is_keyword(line):
            in_nodes = False
            continue
        if in_part and in_nodes and stripped and not stripped.startswith("**"):
            parts = split_data_line(line)
            if len(parts) >= 4:
                x = float(parts[1]) * radius_scale
                y = float(parts[2]) * radius_scale
                z = float(parts[3]) * z_scale
                lines[index] = format_node(parts[0], x, y, z)


def patch_instance_translation(lines, instance_name, z_value):
    waiting_for_translation = False
    marker = "*instance, name=%s" % instance_name.lower()
    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith(marker):
            waiting_for_translation = True
            continue
        if waiting_for_translation:
            if stripped and not is_keyword(line):
                lines[index] = "          0.,           0.,          %s\n" % abaqus_number(z_value)
                return
            if is_keyword(line):
                return
    raise RuntimeError("Instance translation not found for %s" % instance_name)


def patch_press_displacement(lines, row):
    penetration = as_float(row["penetration_m"])
    step_index = find_keyword(lines, "*Step, name=Einpressen, nlgeom=YES")
    if step_index < 0:
        raise RuntimeError("Einpressen step not found")
    end_index = find_keyword(lines, "*End Step", step_index)
    if end_index < 0:
        raise RuntimeError("Einpressen end step not found")
    index = find_line(
        lines,
        lambda line: line.strip().lower().startswith("presse-1.presse_rp, 3, 3"),
        step_index,
        end_index,
    )
    if index < 0:
        raise RuntimeError("Einpressen press displacement line not found")
    lines[index] = "Presse-1.Presse_RP, 3, 3, -%s\n" % abaqus_number(penetration)


def patch_pile_and_press_geometry(lines, row):
    D_m = as_float(row.get("D_m"), REFERENCE_D_M)
    L_m = as_float(row.get("L_m"), REFERENCE_L_M)
    radius_scale = D_m / REFERENCE_D_M
    length_scale = L_m / REFERENCE_L_M
    patch_part_nodes(lines, "Pile", radius_scale=radius_scale, z_scale=length_scale)
    patch_part_nodes(lines, "Presse", radius_scale=radius_scale, z_scale=1.0)
    patch_instance_translation(lines, "Pile-1", REFERENCE_PILE_INSTANCE_Z)
    patch_instance_translation(lines, "Presse-1", REFERENCE_PILE_INSTANCE_Z + L_m)
    patch_press_displacement(lines, row)


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


def patch_steel_material(lines, row):
    start, stop = material_block_bounds(lines, "Stahl")
    density = STEEL_BASE_DENSITY * as_float(row["S"])
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
    # Save nine field-output intervals over the Einpressen step.
    field_interval = step_time / 9.0
    if abs(step_time - 9.0) < 1.0e-9:
        return field_interval, 0.01
    return field_interval, step_time / 1000.0


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


def patch_mc_soil_material(lines, row):
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


def hypoplastic_material_lines(row):
    density = SOIL_BASE_DENSITY * as_float(row["S"])
    return [
        "*Material, name=HYPO-VW96-Sand\n",
        "*Density\n",
        " %s,\n" % abaqus_number(density),
        "*Depvar\n",
        "     20,\n",
        "*User Material, constants=16\n",
        " 0.549779,      0.3,  1.3e+06,    0.324,     0.49,     0.76,     0.86,      0.4\n",
        "       1.,       2.,       5.,   0.0001,      0.7,       1.,       0., 0.615854\n",
    ]


def patch_hypoplastic_soil_material(lines, row):
    start, stop = material_block_bounds(lines, "HYPO-VW96-Sand")
    lines[start:stop] = hypoplastic_material_lines(row)


def remove_solution_initial_conditions(lines):
    index = find_line(
        lines,
        lambda line: line.strip().lower() == "*initial conditions, type=solution",
    )
    if index < 0:
        return
    stop = len(lines)
    for next_index in range(index + 1, len(lines)):
        stripped = lines[next_index].strip()
        if stripped.startswith("*") or stripped.startswith("**"):
            stop = next_index
            break
    del lines[index:stop]


def zero_state_tail():
    return [
        "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n",
        "     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n",
        "     0.0, 0.0, 0.0, 0.0, 0.0\n",
    ]


def solution_initial_condition_lines(row):
    soil_void = abaqus_number(as_float(row["void_ratio_soil_void"]))
    upper = abaqus_number(as_float(row["void_ratio_upper_layer"]))
    down = abaqus_number(as_float(row["void_ratio_down_layer"]))
    lines = ["*Initial Conditions, type=SOLUTION\n"]
    lines.append("    Soil-1.Soil_Void, %s, " % soil_void + zero_state_tail()[0])
    lines.extend(zero_state_tail()[1:])
    lines.append("    Soil-1.Soil_Upper_Layer, %s, " % upper + zero_state_tail()[0])
    lines.extend(zero_state_tail()[1:])
    lines.append("    Soil-1.Soil_down_Layer, %s, " % down + zero_state_tail()[0])
    lines.extend(zero_state_tail()[1:])
    return lines


def patch_solution_initial_conditions(lines, row):
    remove_solution_initial_conditions(lines)
    vf_index = find_line(
        lines,
        lambda line: line.strip().lower()
        == "soil-1.soil_all_geosta, soil-1.imathypo-vw96-sand, 1.",
    )
    if vf_index < 0:
        raise RuntimeError("Volume-fraction material assignment line not found")
    lines[vf_index + 1:vf_index + 1] = solution_initial_condition_lines(row)


def patch_common_template(lines, row):
    patch_steel_material(lines, row)
    patch_gravity(lines, row)
    patch_velocity_timing(lines, row)
    patch_einpressen_output_intervals(lines, row)


def patch_template(template_lines, row):
    lines = list(template_lines)
    soil_model = row.get("soil_model", "")
    patch_pile_and_press_geometry(lines, row)
    if soil_model == "Mohr-Coulomb":
        patch_mc_soil_material(lines, row)
    elif soil_model == "Hypoplastisch":
        patch_hypoplastic_soil_material(lines, row)
        patch_solution_initial_conditions(lines, row)
    else:
        raise RuntimeError("Unsupported soil_model for %s: %s" % (row.get("run_id"), soil_model))
    patch_common_template(lines, row)
    return lines


def verify_reference(lines):
    required = [
        "*Material, name=HYPO-VW96-Sand",
        "*Material, name=Stahl",
        "*Amplitude, name=Amp_Einpressen",
        "*Step, name=Einpressen, nlgeom=YES",
    ]
    text = "".join(lines).lower()
    for keyword in required:
        if keyword.lower() not in text:
            raise RuntimeError("Reference input is missing required keyword: %s" % keyword)
    forbidden = ["*User Material", "*Depvar", "*Initial Conditions, type=SOLUTION"]
    for keyword in forbidden:
        if keyword.lower() in text:
            raise RuntimeError(
                "Reference input must be the Mohr-Coulomb baseline; found %s" % keyword
            )


def remove_if_exists(path):
    if path and os.path.exists(path):
        os.remove(path)


def is_inside(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    return path == root or path.startswith(root + os.sep)


def clean_run_dir(run_dir, run_id):
    run_dir = os.path.abspath(run_dir)
    runs_root = project_path("runs")
    if not is_inside(run_dir, runs_root):
        raise RuntimeError("Refusing to clean outside runs folder: %s" % run_dir)
    if os.path.basename(run_dir) != run_id:
        raise RuntimeError("Refusing to clean run folder with unexpected name: %s" % run_dir)
    if not os.path.isdir(run_dir):
        mkdir_p(run_dir)
        return 0
    removable = set([
        run_id + ".inp",
        "case_config.json",
        "generate_input.ps1",
        "abaqus_input_generation.log",
    ])
    removed = 0
    for name in os.listdir(run_dir):
        path = os.path.join(run_dir, name)
        if name in removable and (os.path.isfile(path) or os.path.islink(path)):
            os.remove(path)
            removed += 1
    return removed


def selected_rows(rows, run_id, limit, soil_model):
    out = []
    for row in rows:
        if run_id and row.get("run_id") != run_id:
            continue
        if soil_model and row.get("soil_model") != soil_model:
            continue
        out.append(row)
        if limit and len(out) >= limit:
            break
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Create Abaqus .inp files by patching a checked reference input file."
    )
    parser.add_argument("--metadata", default=project_path("data", "extracted", "run_metadata_full.csv"))
    parser.add_argument(
        "--reference-inp",
        default=project_path("reference_inputs", "CPT_90_MCM_einpressen_Voll_S001.inp"),
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--soil-model", choices=["Hypoplastisch", "Mohr-Coulomb"], default="")
    parser.add_argument("--clean", action="store_true", help="Remove old files from selected run folders first.")
    parser.add_argument(
        "--clean-derived-data",
        action="store_true",
        help="Remove stale extracted and resampled CSVs referenced by the selected metadata rows.",
    )
    parser.add_argument(
        "--vumat",
        default=project_path("abaqus", "vumat-hypo-2020-hst.for"),
        help="VUMAT file checked when hypoplastic rows are selected.",
    )
    args = parser.parse_args()

    metadata_path = resolve_project_path(args.metadata)
    reference_inp = resolve_project_path(args.reference_inp)
    vumat = resolve_project_path(args.vumat)

    rows = selected_rows(read_csv(metadata_path), args.run_id, args.limit, args.soil_model)
    if not rows:
        raise SystemExit("No rows selected from %s" % metadata_path)
    if not os.path.exists(reference_inp):
        raise SystemExit("Reference input not found: %s" % reference_inp)
    if any(row.get("soil_model") == "Hypoplastisch" for row in rows) and not os.path.exists(vumat):
        raise SystemExit("VUMAT file not found: %s" % vumat)

    with open(reference_inp, "r") as handle:
        template_lines = handle.readlines()
    verify_reference(template_lines)

    removed_files = 0
    written = 0
    by_model = {}
    for row in rows:
        run_id = row["run_id"]
        run_dir = row["run_dir"]
        if args.clean:
            removed_files += clean_run_dir(run_dir, run_id)
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
        by_model[row.get("soil_model", "")] = by_model.get(row.get("soil_model", ""), 0) + 1

    print("Reference: %s" % reference_inp)
    if any(row.get("soil_model") == "Hypoplastisch" for row in rows):
        print("VUMAT: %s" % vumat)
    print("Removed %d old run-folder file(s)." % removed_files)
    for soil_model in sorted(by_model):
        print("Wrote %d %s input file(s)." % (by_model[soil_model], soil_model))
    print("Wrote %d total input file(s)." % written)


if __name__ == "__main__":
    main()

"""
Generate one reference-input variant per planned pile geometry.

This script patches the checked MCM reference input directly. It intentionally
keeps the soil mesh and contact definitions unchanged, and changes the pile and
press diameters plus the assembly/press kinematics needed to keep the pile top
tied to the press and the pile tip at the original starting elevation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_REFERENCE = PROJECT_ROOT / "reference_inputs" / "CPT_90_MCM_einpressen_Voll_S001.inp"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "matrix_full.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reference_inputs" / "generated_pile_geometries"

REFERENCE_D_M = 0.60
REFERENCE_L_M = 12.0


def fmt(value: float) -> str:
    return f"{value:.12g}"


def parse_csv_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def format_node(node_id: str, x: float, y: float, z: float) -> str:
    return f"{int(node_id):7d}, {fmt(x):>16}, {fmt(y):>16}, {fmt(z):>16}"


def is_keyword(line: str) -> bool:
    return line.lstrip().startswith("*")


def planned_geometries(config_path: Path) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = []
    for geometry in config["geometries"]:
        D_m = float(geometry["D_m"])
        L_m = float(geometry["L_m"])
        penetration_m = float(geometry["penetration_m"])
        out.append(
            {
                "geometry_id": geometry["geometry_id"],
                "D_m": D_m,
                "L_m": L_m,
                "penetration_m": penetration_m,
                "penetration_over_D": penetration_m / D_m,
                "penetration_over_L": penetration_m / L_m,
            }
        )
    return out


def patch_part_nodes(
    lines: list[str],
    part_name: str,
    radius_scale: float,
    z_scale: float = 1.0,
) -> list[str]:
    out = []
    in_part = False
    in_nodes = False
    part_marker = "name=%s" % part_name.lower()

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("*part") and part_marker in lower:
            in_part = True
            in_nodes = False
            out.append(line)
            continue
        if in_part and lower.startswith("*end part"):
            in_part = False
            in_nodes = False
            out.append(line)
            continue
        if in_part and lower.startswith("*node"):
            in_nodes = True
            out.append(line)
            continue
        if in_part and in_nodes and is_keyword(line):
            in_nodes = False
            out.append(line)
            continue
        if in_part and in_nodes and stripped and not stripped.startswith("**"):
            parts = parse_csv_line(line)
            if len(parts) >= 4:
                node_id = parts[0]
                x = float(parts[1]) * radius_scale
                y = float(parts[2]) * radius_scale
                z = float(parts[3]) * z_scale
                out.append(format_node(node_id, x, y, z))
                continue
        out.append(line)
    return out


def patch_pile_and_press_nodes(lines: list[str], D_m: float, L_m: float) -> list[str]:
    radius_scale = D_m / REFERENCE_D_M
    length_scale = L_m / REFERENCE_L_M
    lines = patch_part_nodes(lines, "Pile", radius_scale=radius_scale, z_scale=length_scale)
    lines = patch_part_nodes(lines, "Presse", radius_scale=radius_scale, z_scale=1.0)
    return lines


def patch_instance_translation(lines: list[str], instance_name: str, z_value: float) -> list[str]:
    out = []
    waiting_for_translation = False
    instance_re = re.compile(r"^\*Instance,\s*name=%s\b" % re.escape(instance_name), re.IGNORECASE)
    for line in lines:
        if instance_re.search(line.strip()):
            waiting_for_translation = True
            out.append(line)
            continue
        if waiting_for_translation:
            if line.strip() and not is_keyword(line):
                out.append(f"          0.,           0.,          {fmt(z_value)}")
                waiting_for_translation = False
                continue
            if is_keyword(line):
                waiting_for_translation = False
        out.append(line)
    return out


def patch_step_kinematics(lines: list[str], penetration_m: float, velocity_m_per_s: float) -> list[str]:
    step_time = penetration_m / max(velocity_m_per_s, 1.0e-12)
    out = []
    after_amp = False
    in_einpressen = False
    waiting_for_dynamic_time = False
    waiting_for_press_u3 = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("*amplitude") and "name=amp_einpressen" in lower:
            out.append(line)
            after_amp = True
            continue
        if after_amp:
            out.append(f"             0.,              0.,              {fmt(step_time)},              1.")
            after_amp = False
            continue

        if lower.startswith("*step") and "name=einpressen" in lower:
            in_einpressen = True
            out.append(line)
            continue
        if in_einpressen and lower.startswith("*end step"):
            in_einpressen = False
            waiting_for_dynamic_time = False
            waiting_for_press_u3 = False
            out.append(line)
            continue
        if in_einpressen and lower.startswith("*dynamic"):
            out.append(line)
            waiting_for_dynamic_time = True
            continue
        if in_einpressen and waiting_for_dynamic_time:
            out.append(f", {fmt(step_time)}")
            waiting_for_dynamic_time = False
            continue
        if in_einpressen and lower.startswith("*boundary") and "amplitude=amp_einpressen" in lower:
            out.append(line)
            waiting_for_press_u3 = True
            continue
        if in_einpressen and waiting_for_press_u3 and stripped.startswith("Presse-1.Presse_RP, 3, 3"):
            out.append(f"Presse-1.Presse_RP, 3, 3, -{fmt(penetration_m)}")
            waiting_for_press_u3 = False
            continue

        out.append(line)

    return out


def add_header(lines: list[str], geometry: dict, velocity_m_per_s: float) -> list[str]:
    header = [
        "**",
        "** Generated pile-geometry variant from CPT_90_MCM_einpressen_Voll_S001.inp",
        f"** geometry_id={geometry['geometry_id']}",
        f"** D_m={fmt(geometry['D_m'])}",
        f"** L_m={fmt(geometry['L_m'])}",
        f"** penetration_m={fmt(geometry['penetration_m'])}",
        f"** penetration_over_D={fmt(geometry['penetration_over_D'])}",
        f"** penetration_over_L={fmt(geometry['penetration_over_L'])}",
        f"** velocity_m_per_s={fmt(velocity_m_per_s)}",
        "** Soil mesh and contact definitions are copied unchanged.",
        "** Pile and Presse diameters are scaled together.",
        "**",
    ]
    return lines[:4] + header + lines[4:]


def patch_geometry(reference_lines: list[str], geometry: dict, velocity_m_per_s: float) -> list[str]:
    pile_tip_z = 27.0
    pile_top_z = pile_tip_z + geometry["L_m"]
    lines = patch_pile_and_press_nodes(reference_lines, geometry["D_m"], geometry["L_m"])
    lines = patch_instance_translation(lines, "Pile-1", pile_tip_z)
    lines = patch_instance_translation(lines, "Presse-1", pile_top_z)
    lines = patch_step_kinematics(lines, geometry["penetration_m"], velocity_m_per_s)
    return add_header(lines, geometry, velocity_m_per_s)


def output_name(geometry: dict) -> str:
    D = str(geometry["D_m"]).replace(".", "p")
    L = str(round(geometry["L_m"], 3)).replace(".", "p")
    P = str(round(geometry["penetration_m"], 3)).replace(".", "p")
    return f"CPT_90_MCM_{geometry['geometry_id']}_D{D}_L{L}_P{P}_S001.inp"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 9 pile-only geometry input variants.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--velocity-m-per-s", type=float, default=0.5)
    args = parser.parse_args()

    reference = Path(args.reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_lines = reference.read_text(encoding="utf-8", errors="ignore").splitlines()
    geometries = planned_geometries(Path(args.config))
    written = []
    for geometry in geometries:
        patched = patch_geometry(reference_lines, geometry, args.velocity_m_per_s)
        output = output_dir / output_name(geometry)
        output.write_text("\n".join(patched) + "\n", encoding="utf-8")
        written.append(output)
        print(f"Wrote {output}")

    print(f"Wrote {len(written)} pile-geometry input file(s) to {output_dir}")


if __name__ == "__main__":
    main()

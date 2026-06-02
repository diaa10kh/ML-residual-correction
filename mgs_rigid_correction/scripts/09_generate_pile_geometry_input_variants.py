"""
Generate scaled reference-input variants for the non-reference pile diameters.

This script patches the checked MCM reference inputs directly. It intentionally
keeps the soil mesh, contact definitions, pile length, assembly translations,
and press displacement unchanged. Only the pile and press diameters are scaled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REFERENCE_INPUTS = PROJECT_ROOT / "reference_inputs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "matrix_full_hypoplastic.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reference_inputs" / "generated_pile_geometries"
DEFAULT_REFERENCE = REFERENCE_INPUTS / "Pile_11_m_einpressen_Voll_S001.inp"

REFERENCE_D_M = 0.60


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


def patch_pile_and_press_nodes(lines: list[str], D_m: float) -> list[str]:
    radius_scale = D_m / REFERENCE_D_M
    lines = patch_part_nodes(lines, "Pile", radius_scale=radius_scale, z_scale=1.0)
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


def add_header(lines: list[str], geometry: dict, velocity_m_per_s: float, source_name: str) -> list[str]:
    header = [
        "**",
        f"** Generated pile-geometry variant from {source_name}",
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


def patch_geometry(
    reference_lines: list[str],
    geometry: dict,
    velocity_m_per_s: float,
    source_name: str,
) -> list[str]:
    lines = patch_pile_and_press_nodes(reference_lines, geometry["D_m"])
    return add_header(lines, geometry, velocity_m_per_s, source_name)


def output_name(geometry: dict) -> str:
    D = int(round(100.0 * geometry["D_m"]))
    return f"CPT_90_MCM_{geometry['geometry_id']}_D{D:03d}_S007.inp"


def reference_path_for_geometry(geometry: dict, reference_path: Path) -> Path:
    return reference_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pile-only geometry input variants from the active matrix.")
    parser.add_argument("--reference-inp", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--velocity-m-per-s", type=float, default=0.5)
    args = parser.parse_args()

    reference_path = Path(args.reference_inp)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("CPT_90_MCM_G*_D*_L*_P*_S*.inp"):
        stale.unlink()
    for stale in output_dir.glob("CPT_90_MCM_G*_D*_S*.inp"):
        stale.unlink()

    reference_cache: dict[Path, list[str]] = {}
    geometries = planned_geometries(Path(args.config))
    written = []
    for geometry in geometries:
        if abs(float(geometry["D_m"]) - REFERENCE_D_M) <= 1.0e-9:
            print(f"Using reference diameter directly for {geometry['geometry_id']} D={fmt(geometry['D_m'])}; no scaled template written")
            continue
        reference = reference_path_for_geometry(geometry, reference_path)
        if reference not in reference_cache:
            reference_cache[reference] = reference.read_text(encoding="utf-8", errors="ignore").splitlines()
        patched = patch_geometry(reference_cache[reference], geometry, args.velocity_m_per_s, reference.name)
        output = output_dir / output_name(geometry)
        output.write_text("\n".join(patched) + "\n", encoding="utf-8")
        written.append(output)
        print(f"Wrote {output}")

    print(f"Wrote {len(written)} pile-geometry input file(s) to {output_dir}")


if __name__ == "__main__":
    main()

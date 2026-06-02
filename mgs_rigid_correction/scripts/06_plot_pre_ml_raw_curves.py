"""
Plot raw extracted curves before ML dataset building or training.

This script intentionally does not resample, pair, remove outliers, or build
training data. It reads the extracted per-run CSV files directly, applies only
a centered moving average, and saves PNGs for visual inspection.

Default output:
  plots/raw_averaged_plots/{mcm,hypoplastic}/overview_by_geometry/
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "extracted" / "per_run_csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "raw_averaged_plots"

# Adjustable plotting parameters.
MCM_MOVING_AVERAGE_WINDOW = 30
HYPOPLASTIC_MOVING_AVERAGE_WINDOW = 30
MCM_QS_XLIM_KPA = None
HYPOPLASTIC_QS_XLIM_KPA = None

DENSITY_ORDER = ["DENS_LOW", "DENS_MED", "DENS_REF", "DENS_HIGH"]
VELOCITY_ORDER = ["V_LOW", "V_REF", "V_HIGH"]

DENSITY_LABELS = {
    "DENS_LOW": "ID=0.3",
    "DENS_MED": "ID=0.6",
    "DENS_REF": "ID=0.8",
    "DENS_HIGH": "ID=0.9",
}

VELOCITY_LABELS = {
    "V_LOW": "v=25 cm/s",
    "V_REF": "v=50 cm/s",
    "V_HIGH": "v=100 cm/s",
}

S_COLORS = {
    1: "#1f2933",
    3: "#1f2933",
    7: "#1f2933",
    10: "#1f77b4",
    15: "#1f77b4",
    30: "#ff7f0e",
    50: "#2ca02c",
    100: "#d62728",
}

SOIL_MODELS = [
    {
        "name": "mcm",
        "label": "MCM",
        "raw_subdir": "mcm",
        "patterns": ["MC_G*.csv"],
        "metadata_path": DATA_ROOT / "extracted" / "run_metadata_full_mohr_coulomb.csv",
        "moving_average_window": MCM_MOVING_AVERAGE_WINDOW,
        "qs_xlim": MCM_QS_XLIM_KPA,
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "raw_subdir": "hypoplastic",
        "patterns": ["G*.csv"],
        "metadata_path": DATA_ROOT / "extracted" / "run_metadata_full.csv",
        "moving_average_window": HYPOPLASTIC_MOVING_AVERAGE_WINDOW,
        "qs_xlim": HYPOPLASTIC_QS_XLIM_KPA,
    },
]

QUANTITIES = {
    "qb_MPa": {
        "label": "Base resistance qb [MPa]",
        "filename": "qb",
        "xlim": None,
    },
    "qs_kPa": {
        "label": "Shaft resistance qs [kPa]",
        "filename": "qs",
        "xlim": None,
    },
}


def normalise_soil_model(value: str) -> str:
    return "mcm" if value == "mohr_coulomb" else value


def parse_scenario_id(scenario_id: str) -> dict[str, str]:
    parts = str(scenario_id).split("_")
    if parts and parts[0] == "MC":
        parts = parts[1:]

    geometry_parts = parts[:1]
    density = ""
    velocity = ""

    try:
        dens_idx = parts.index("DENS")
        geometry_parts = parts[:dens_idx]
        if dens_idx + 1 < len(parts):
            density = f"DENS_{parts[dens_idx + 1]}"
    except ValueError:
        dens_idx = -1

    try:
        vel_idx = parts.index("V", max(dens_idx, 0))
        if vel_idx + 1 < len(parts):
            velocity = f"V_{parts[vel_idx + 1]}"
    except ValueError:
        pass

    geometry_label = "_".join(geometry_parts) if geometry_parts else ""
    geometry_id = geometry_parts[0] if geometry_parts else geometry_label
    return {
        "geometry_label": geometry_label,
        "geometry_id": geometry_id,
        "density": density,
        "velocity": velocity,
    }


def scenario_sort_key(scenario_id: str):
    parsed = parse_scenario_id(scenario_id)
    density = parsed["density"]
    velocity = parsed["velocity"]
    try:
        density_idx = DENSITY_ORDER.index(density)
    except ValueError:
        density_idx = len(DENSITY_ORDER)
    try:
        velocity_idx = VELOCITY_ORDER.index(velocity)
    except ValueError:
        velocity_idx = len(VELOCITY_ORDER)
    return geometry_sort_key(parsed["geometry_label"]), density_idx, velocity_idx, scenario_id


def geometry_sort_key(geometry_label: str):
    match = re.match(r"^G(\d+)(?:_|$)", str(geometry_label))
    if match:
        return int(match.group(1)), str(geometry_label)
    return 10_000, str(geometry_label)


def as_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def discover_csv_paths(raw_dir: Path, patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(raw_dir.glob(pattern))
    return sorted(set(paths))


def model_raw_dir(raw_dir: Path, run: dict) -> Path:
    nested = raw_dir / run["raw_subdir"]
    return nested if nested.is_dir() else raw_dir


def load_scenario_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    if not metadata_path.exists():
        return {}

    out = {}
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            scenario_id = row.get("scenario_id", "")
            if scenario_id and scenario_id not in out:
                out[scenario_id] = row
    return out


def expected_run_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()

    out = set()
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("run_id", "")
            if run_id:
                out.add(run_id)
    return out


def paths_in_metadata(paths: list[Path], run: dict) -> list[Path]:
    expected = expected_run_ids(run["metadata_path"])
    if not expected:
        return paths
    return [path for path in paths if path.stem in expected]


def report_metadata_coverage(paths: list[Path], run: dict):
    expected = expected_run_ids(run["metadata_path"])
    if not expected:
        print(f"WARNING: no metadata file found for {run['label']} at {run['metadata_path']}")
        return

    discovered = {path.stem for path in paths}
    observed = expected.intersection(discovered)
    missing = sorted(expected.difference(observed))
    extra_count = len(discovered.difference(expected))
    print(
        f"Metadata coverage for {run['label']}: {len(observed)} observed / "
        f"{len(expected)} expected CSV files"
    )
    if missing:
        print(f"  WARNING: {len(missing)} expected CSV file(s) are missing")
        for run_id in missing[:8]:
            print(f"    missing: {run_id}.csv")
        if len(missing) > 8:
            print(f"    ... {len(missing) - 8} more")
    if extra_count:
        print(f"  Ignoring {extra_count} CSV file(s) not listed in the selected metadata")


def quantity_xlim(quantity: str, soil_run):
    if quantity == "qs_kPa":
        return soil_run["qs_xlim"]
    return QUANTITIES[quantity]["xlim"]


def qs_xlim_label(soil_run) -> str:
    xlim = soil_run["qs_xlim"]
    if xlim is None:
        return "auto"
    return f"{xlim[0]}..{xlim[1]} kPa"


def load_curves(paths: list[Path], label: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No non-empty {label} CSV files were found")

    df = pd.concat(frames, ignore_index=True)
    required = {"run_id", "scenario_id", "S", "z_m", "qb_MPa"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{label} CSV files are missing required column(s): {', '.join(missing)}")

    for column in ["S", "z_m", "qb_MPa", "qs_kPa"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["scenario_id", "S", "z_m"]).copy()
    df["S_int"] = df["S"].astype(int)

    parsed_by_scenario = {
        scenario_id: parse_scenario_id(scenario_id)
        for scenario_id in df["scenario_id"].dropna().unique()
    }
    for column in ["geometry_label", "geometry_id", "density", "velocity"]:
        df[column] = df["scenario_id"].map(
            lambda scenario_id, key=column: parsed_by_scenario[str(scenario_id)][key]
        )

    return df


def plot_run_quantity(ax, run: pd.DataFrame, quantity: str, window: int, s_value: int, reference_s: int):
    run = run.sort_values("z_m")
    run = run[run["z_m"] > 0.0]
    run = run.dropna(subset=["z_m", quantity])
    if run.empty:
        return

    is_reference = s_value == reference_s
    label = f"S={reference_s} reference" if is_reference else f"S={s_value}"
    ax.plot(
        smooth(run[quantity], window),
        run["z_m"],
        color=S_COLORS.get(s_value),
        linewidth=1.9 if is_reference else 1.25,
        linestyle="-" if is_reference else "--",
        alpha=0.95,
        label=label,
    )


def geometry_title(geometry_label: str, geometry_df: pd.DataFrame, scenario_metadata: dict[str, dict[str, str]]) -> str:
    scenario_ids = geometry_df["scenario_id"].dropna().unique()
    meta = {}
    for scenario_id in scenario_ids:
        meta = scenario_metadata.get(str(scenario_id), {})
        if meta:
            break

    geometry_id = meta.get("geometry_id") or geometry_df["geometry_id"].dropna().iloc[0]
    D_m = as_float(meta.get("D_m"))

    if D_m is None:
        match = re.match(r"^(G\d+)_D(\d+)(?:_P\d+)?$", str(geometry_label))
        if match:
            geometry_id = geometry_id or match.group(1)
            D_m = D_m if D_m is not None else int(match.group(2)) / 100.0

    if D_m is not None:
        return f"{geometry_id} (D={D_m:g} m)"
    return str(geometry_label)


def legend_from_axes(axes_flat):
    by_label = {}
    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label.setdefault(label, handle)
    return list(by_label.values()), list(by_label.keys())


def plot_geometry_overviews(
    df: pd.DataFrame,
    soil_run,
    quantity: str,
    output_dir: Path,
    window: int,
    scenario_metadata: dict[str, dict[str, str]],
    reference_s: int,
):
    soil_label = soil_run["label"]
    soil_name = soil_run["name"]
    geometry_labels = sorted(df["geometry_label"].dropna().unique(), key=geometry_sort_key)

    overview_dir = output_dir / soil_name / "overview_by_geometry"
    overview_dir.mkdir(parents=True, exist_ok=True)

    for geometry_label in geometry_labels:
        geometry_df = df[df["geometry_label"] == geometry_label]
        scenario_ids = sorted(geometry_df["scenario_id"].dropna().unique(), key=scenario_sort_key)
        if not scenario_ids:
            continue

        fig, axes = plt.subplots(4, 3, figsize=(13.5, 16), sharey=True)
        axes_flat = axes.flatten()
        plotted_axes = set()
        max_depth = geometry_df["z_m"].max()

        for scenario_id in scenario_ids:
            scenario = geometry_df[geometry_df["scenario_id"] == scenario_id]
            parsed = parse_scenario_id(scenario_id)
            density = parsed["density"]
            velocity = parsed["velocity"]

            try:
                row_idx = DENSITY_ORDER.index(density)
                col_idx = VELOCITY_ORDER.index(velocity)
            except ValueError:
                continue

            ax = axes[row_idx, col_idx]
            plotted_axes.add(ax)
            for s_value in sorted(scenario["S_int"].dropna().unique()):
                run = scenario[scenario["S_int"] == s_value]
                plot_run_quantity(ax, run, quantity, window, int(s_value), reference_s)

            title = f"{DENSITY_LABELS.get(density, density)}, {VELOCITY_LABELS.get(velocity, velocity)}"
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
            ax.set_ylim(float(max_depth) * 1.02, 0.0)
            xlim = quantity_xlim(quantity, soil_run)
            if xlim is not None:
                ax.set_xlim(*xlim)

        for ax in axes_flat:
            if ax not in plotted_axes:
                ax.set_visible(False)

        for ax in axes[:, 0]:
            ax.set_ylabel("Penetration depth z [m]")
        for ax in axes[-1, :]:
            ax.set_xlabel(QUANTITIES[quantity]["label"])

        handles, labels = legend_from_axes(axes_flat)
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
        fig.suptitle(
            f"{soil_label} raw extracted curves: {geometry_title(geometry_label, geometry_df, scenario_metadata)}\n"
            f"{window}-point moving average only",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))

        out = (
            overview_dir
            / f"{soil_name}_{safe_filename(geometry_label)}_overview_"
            f"{QUANTITIES[quantity]['filename']}_{window}pt_average.png"
        )
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print(f"Wrote {out}")


def plot_scenarios(
    df: pd.DataFrame,
    soil_run,
    output_dir: Path,
    window: int,
    quantities,
    scenario_limit: int | None,
    scenario_metadata: dict[str, dict[str, str]],
    reference_s: int,
):
    soil_label = soil_run["label"]
    soil_name = soil_run["name"]
    scenario_ids = sorted(df["scenario_id"].dropna().unique(), key=scenario_sort_key)
    total_scenarios = len(scenario_ids)
    if scenario_limit is not None:
        scenario_ids = scenario_ids[: max(0, scenario_limit)]
        if len(scenario_ids) < total_scenarios:
            print(f"Writing {len(scenario_ids)} of {total_scenarios} detailed scenario plot(s)")
    scenario_dir = output_dir / soil_name / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in scenario_ids:
        available_quantities = [q for q in quantities if q in df.columns]
        if not available_quantities:
            continue

        fig, axes = plt.subplots(1, len(available_quantities), figsize=(7 * len(available_quantities), 6), sharey=True)
        axes = np.ravel(axes)
        scenario = df[df["scenario_id"] == scenario_id]
        parsed = parse_scenario_id(scenario_id)
        density = parsed["density"]
        velocity = parsed["velocity"]
        scenario_meta = scenario_metadata.get(str(scenario_id), {})
        D_m = as_float(scenario_meta.get("D_m"))
        geometry_text = parsed["geometry_label"]
        if D_m is not None:
            geometry_text = f"{scenario_meta.get('geometry_id') or parsed['geometry_id']} (D={D_m:g} m)"

        for ax, quantity in zip(axes, available_quantities):
            for s_value in sorted(scenario["S_int"].dropna().unique()):
                run = scenario[scenario["S_int"] == s_value]
                plot_run_quantity(ax, run, quantity, window, int(s_value), reference_s)

            ax.set_xlabel(QUANTITIES[quantity]["label"])
            ax.set_ylim(float(scenario["z_m"].max()) * 1.02, 0.0)
            xlim = quantity_xlim(quantity, soil_run)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Penetration depth z [m]")
        fig.suptitle(
            f"{soil_label}: {geometry_text}, {DENSITY_LABELS.get(density, density)}, "
            f"{VELOCITY_LABELS.get(velocity, velocity)}\n"
            f"{window}-point moving average only",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))

        out = scenario_dir / f"{safe_filename(scenario_id)}_{window}pt_average.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        print(f"Wrote {out}")


def filter_geometries(df: pd.DataFrame, geometry_filters: list[str]) -> pd.DataFrame:
    if not geometry_filters:
        return df

    wanted = {item.strip() for item in geometry_filters if item.strip()}
    if not wanted:
        return df

    filtered = df[df["geometry_id"].isin(wanted) | df["geometry_label"].isin(wanted)].copy()
    if filtered.empty:
        raise SystemExit(f"No runs matched --geometry {', '.join(sorted(wanted))}")
    return filtered


def plot_soil_model(
    run,
    raw_dir: Path,
    output_dir: Path,
    window_override: int | None,
    quantities,
    geometry_filters: list[str],
    plot_details: bool,
    scenario_limit: int | None,
    allow_missing: bool,
):
    input_dir = model_raw_dir(raw_dir, run)
    paths = discover_csv_paths(input_dir, run["patterns"])
    if not paths:
        message = f"No {run['label']} CSV files found in {input_dir} with patterns {', '.join(run['patterns'])}"
        if allow_missing:
            print(f"WARNING: {message}; skipping")
            return
        raise SystemExit(message)

    report_metadata_coverage(paths, run)
    paths = paths_in_metadata(paths, run)
    if not paths:
        raise SystemExit(f"No {run['label']} CSV files matched the selected metadata")
    scenario_metadata = load_scenario_metadata(run["metadata_path"])
    df = load_curves(paths, run["label"])
    df = filter_geometries(df, geometry_filters)
    window = window_override if window_override is not None else run["moving_average_window"]
    scenario_count = df["scenario_id"].nunique()
    run_count = df["run_id"].nunique()
    geometry_count = df["geometry_label"].nunique()
    reference_s = int(df["S_int"].min())
    print(
        f"Loaded {run_count} {run['label']} runs in {scenario_count} scenarios "
        f"across {geometry_count} geometries from {input_dir}; "
        f"moving average window={window}, qs_xlim={qs_xlim_label(run)}, reference S={reference_s}"
    )

    selected_quantities = [q for q in quantities if q in df.columns]
    for quantity in selected_quantities:
        plot_geometry_overviews(df, run, quantity, output_dir, window, scenario_metadata, reference_s)
    if plot_details:
        plot_scenarios(df, run, output_dir, window, selected_quantities, scenario_limit, scenario_metadata, reference_s)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw extracted curves before ML dataset building or training. "
            "The default output is compact: one 12-panel overview page per geometry."
        )
    )
    parser.add_argument(
        "--soil-model",
        choices=["all", "mcm", "mohr_coulomb", "hypoplastic"],
        default="all",
        help="'mohr_coulomb' is accepted as an alias for 'mcm'.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DIR),
        help="Folder containing extracted per-run CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for PNG plots.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override the model-specific moving-average window in points.",
    )
    parser.add_argument(
        "--quantity",
        choices=["both", "qb", "qs"],
        default="both",
        help="Quantity to plot.",
    )
    parser.add_argument(
        "--geometry",
        nargs="*",
        default=[],
        help="Optional geometry IDs/labels to plot, e.g. G0 or G0_D060. Defaults to all available geometries.",
    )
    parser.add_argument(
        "--plot-scenarios",
        action="store_true",
        help="Also write one detailed PNG per scenario. Off by default to avoid hundreds of plots.",
    )
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=None,
        help="Maximum number of detailed scenario PNGs when --plot-scenarios is used.",
    )
    args = parser.parse_args()

    selected = normalise_soil_model(args.soil_model)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if args.quantity == "both":
        quantities = ["qb_MPa", "qs_kPa"]
    elif args.quantity == "qb":
        quantities = ["qb_MPa"]
    else:
        quantities = ["qs_kPa"]

    for run in SOIL_MODELS:
        if selected != "all" and run["name"] != selected:
            continue
        plot_soil_model(
            run,
            raw_dir,
            output_dir,
            args.window,
            quantities,
            args.geometry,
            args.plot_scenarios,
            args.scenario_limit,
            allow_missing=selected == "all",
        )


if __name__ == "__main__":
    main()

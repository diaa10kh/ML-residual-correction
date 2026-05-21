"""
Plot raw extracted curves before ML dataset building or training.

This script intentionally does not resample, pair, remove outliers, or build
training data. It reads the extracted per-run CSV files directly, applies only
a centered moving average, and saves PNGs for visual inspection.

Default output:
  plots/raw_averaged_plots/{mcm,hypoplastic}/
"""

from __future__ import annotations

import argparse
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
RAW_DIR = PROJECT_ROOT / "data" / "extracted" / "per_run_csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "raw_averaged_plots"

# Adjustable plotting parameters.
MCM_MOVING_AVERAGE_WINDOW = 30
HYPOPLASTIC_MOVING_AVERAGE_WINDOW = 30
MCM_QS_XLIM_KPA = (0.0, 200.0)
HYPOPLASTIC_QS_XLIM_KPA = (0.0, 30.0)

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
    10: "#1f77b4",
    30: "#ff7f0e",
    50: "#2ca02c",
    100: "#d62728",
}

SOIL_MODELS = [
    {
        "name": "mcm",
        "label": "MCM",
        "pattern": "MC_G0_*.csv",
        "moving_average_window": MCM_MOVING_AVERAGE_WINDOW,
        "qs_xlim": MCM_QS_XLIM_KPA,
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "pattern": "G0_*.csv",
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


def parse_scenario_id(scenario_id: str):
    parts = scenario_id.split("_")
    if parts[0] == "MC":
        parts = parts[1:]
    if len(parts) < 5:
        return "", ""
    density = "_".join(parts[1:3])
    velocity = "_".join(parts[3:5])
    return density, velocity


def scenario_sort_key(scenario_id: str):
    density, velocity = parse_scenario_id(scenario_id)
    try:
        density_idx = DENSITY_ORDER.index(density)
    except ValueError:
        density_idx = len(DENSITY_ORDER)
    try:
        velocity_idx = VELOCITY_ORDER.index(velocity)
    except ValueError:
        velocity_idx = len(VELOCITY_ORDER)
    return density_idx, velocity_idx, scenario_id


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def quantity_xlim(quantity: str, soil_run):
    if quantity == "qs_kPa":
        return soil_run["qs_xlim"]
    return QUANTITIES[quantity]["xlim"]


def load_curves(raw_dir: Path, pattern: str, label: str) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No {label} CSV files found in {raw_dir} with pattern {pattern}")

    df = pd.concat(frames, ignore_index=True)
    required = {"run_id", "scenario_id", "S", "z_m", "qb_MPa"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{label} CSV files are missing required column(s): {', '.join(missing)}")

    for column in ["S", "z_m", "qb_MPa", "qs_kPa"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["scenario_id", "S", "z_m"])


def plot_run_quantity(ax, run: pd.DataFrame, quantity: str, window: int, s_value: int):
    run = run.sort_values("z_m")
    run = run[run["z_m"] > 0.0]
    run = run.dropna(subset=["z_m", quantity])
    if run.empty:
        return

    label = "S=1 reference" if s_value == 1 else f"S={s_value}"
    ax.plot(
        smooth(run[quantity], window),
        run["z_m"],
        color=S_COLORS.get(s_value),
        linewidth=1.9 if s_value == 1 else 1.25,
        linestyle="-" if s_value == 1 else "--",
        alpha=0.95,
        label=label,
    )


def plot_overview(df: pd.DataFrame, soil_run, quantity: str, output_dir: Path, window: int):
    soil_label = soil_run["label"]
    soil_name = soil_run["name"]
    scenario_ids = sorted(df["scenario_id"].dropna().unique(), key=scenario_sort_key)
    if not scenario_ids:
        return

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 16), sharey=True)
    axes_flat = axes.flatten()

    for ax, scenario_id in zip(axes_flat, scenario_ids):
        scenario = df[df["scenario_id"] == scenario_id]
        density, velocity = parse_scenario_id(scenario_id)

        for s_value in sorted(scenario["S"].dropna().astype(int).unique()):
            run = scenario[scenario["S"].astype(int) == s_value]
            plot_run_quantity(ax, run, quantity, window, s_value)

        title = f"{DENSITY_LABELS.get(density, density)}, {VELOCITY_LABELS.get(velocity, velocity)}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(9.0, 0.0)
        xlim = quantity_xlim(quantity, soil_run)
        if xlim is not None:
            ax.set_xlim(*xlim)

    for ax in axes_flat[len(scenario_ids):]:
        ax.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Penetration depth z [m]")
    for ax in axes[-1, :]:
        ax.set_xlabel(QUANTITIES[quantity]["label"])

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{soil_label} raw extracted curves, {window}-point moving average only",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out = output_dir / soil_name / f"{soil_name}_overview_{QUANTITIES[quantity]['filename']}_{window}pt_average.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_scenarios(df: pd.DataFrame, soil_run, output_dir: Path, window: int, quantities):
    soil_label = soil_run["label"]
    soil_name = soil_run["name"]
    scenario_ids = sorted(df["scenario_id"].dropna().unique(), key=scenario_sort_key)
    scenario_dir = output_dir / soil_name / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for scenario_id in scenario_ids:
        available_quantities = [q for q in quantities if q in df.columns]
        if not available_quantities:
            continue

        fig, axes = plt.subplots(1, len(available_quantities), figsize=(7 * len(available_quantities), 6), sharey=True)
        axes = np.ravel(axes)
        scenario = df[df["scenario_id"] == scenario_id]
        density, velocity = parse_scenario_id(scenario_id)

        for ax, quantity in zip(axes, available_quantities):
            for s_value in sorted(scenario["S"].dropna().astype(int).unique()):
                run = scenario[scenario["S"].astype(int) == s_value]
                plot_run_quantity(ax, run, quantity, window, s_value)

            ax.set_xlabel(QUANTITIES[quantity]["label"])
            ax.set_ylim(9.0, 0.0)
            xlim = quantity_xlim(quantity, soil_run)
            if xlim is not None:
                ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Penetration depth z [m]")
        fig.suptitle(
            f"{soil_label}: {DENSITY_LABELS.get(density, density)}, "
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


def plot_soil_model(run, raw_dir: Path, output_dir: Path, window_override: int | None, quantities):
    df = load_curves(raw_dir, run["pattern"], run["label"])
    window = window_override if window_override is not None else run["moving_average_window"]
    scenario_count = df["scenario_id"].nunique()
    run_count = df["run_id"].nunique()
    print(
        f"Loaded {run_count} {run['label']} runs in {scenario_count} scenarios from {raw_dir}; "
        f"moving average window={window}, qs_xlim={run['qs_xlim']} kPa"
    )

    selected_quantities = [q for q in quantities if q in df.columns]
    for quantity in selected_quantities:
        plot_overview(df, run, quantity, output_dir, window)
    plot_scenarios(df, run, output_dir, window, selected_quantities)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot raw extracted curves for the 12 density/velocity scenarios "
            "before ML dataset building or training."
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
        help="Quantity to plot in overview PNGs. Scenario PNGs include the selected quantity/quantities.",
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
        plot_soil_model(run, raw_dir, output_dir, args.window, quantities)


if __name__ == "__main__":
    main()

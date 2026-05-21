"""
Build ML residual datasets from extracted Abaqus CSV files.

This is the repository-adapted version of the student dataset pipeline.  It
keeps the repo input layout, but writes separate outputs for MCM and
hypoplastic runs:

  Input:  data/extracted/per_run_csv/{run_id}.csv
  Output: data/processed/ml/{mcm,hypoplastic}/real_dataset.csv
          data/processed/ml/{mcm,hypoplastic}/real_dataset_plot.csv
          data/processed/ml/{mcm,hypoplastic}/dataset_meta.json

The current Phase 0 grids are expected to contain 60 extracted CSVs per soil
model: 4 densities x 3 velocities x 5 scaling factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "extracted" / "per_run_csv"
OUT_ROOT = DATA_ROOT / "processed" / "ml"

S_REF = 1
EXPECTED_CSV_COUNT = 60

DENS_MAP = {"LOW": 0.30, "MED": 0.60, "REF": 0.80, "HIGH": 0.90}
VPEN_MAP = {"LOW": 25.0, "REF": 50.0, "HIGH": 100.0}

SMOOTH_WINDOW_QB = 30
SMOOTH_WINDOW_QS = 30
MIN_TRAINING_DEPTH = 1.0
OUTLIER_STD_THRESH = 3.0
GRADIENT_WINDOW = 5

SOIL_MODEL_RUNS = [
    {
        "name": "mcm",
        "label": "MCM",
        "patterns": ["MC_*.csv"],
        "metadata_path": DATA_ROOT / "extracted" / "run_metadata_phase0_mohr_coulomb.csv",
        "out_dir": OUT_ROOT / "mcm",
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "patterns": ["G0_*.csv"],
        "metadata_path": DATA_ROOT / "extracted" / "run_metadata_phase0.csv",
        "out_dir": OUT_ROOT / "hypoplastic",
    },
]


def normalise_soil_model(value: str) -> str:
    if value == "mohr_coulomb":
        return "mcm"
    return value


def smooth_curve(series, window: int):
    return (
        pd.Series(series)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .values
    )


def compute_depth_gradient(values, depths, window: int = GRADIENT_WINDOW):
    vals = np.asarray(values, dtype=float)
    dpts = np.asarray(depths, dtype=float)
    grad = np.full_like(vals, np.nan)

    for i in range(1, len(vals) - 1):
        dz = dpts[i + 1] - dpts[i - 1]
        if dz > 0:
            grad[i] = (vals[i + 1] - vals[i - 1]) / dz

    if len(vals) > 1:
        dz0 = dpts[1] - dpts[0]
        grad[0] = (vals[1] - vals[0]) / dz0 if dz0 > 0 else 0.0
        dz_last = dpts[-1] - dpts[-2]
        grad[-1] = (vals[-1] - vals[-2]) / dz_last if dz_last > 0 else 0.0

    return (
        pd.Series(grad)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .bfill()
        .ffill()
        .fillna(0.0)
        .values
    )


def parse_filename(path: Path):
    base = path.stem
    parts = base.split("_")
    try:
        dens_idx = parts.index("DENS")
        dens_level = parts[dens_idx + 1]
        v_idx = parts.index("V")
        v_level = parts[v_idx + 1]
        s_part = [part for part in parts if part.startswith("S") and part[1:].isdigit()]
        s_value = int(s_part[0][1:]) if s_part else None
        scenario_id = "_".join(parts[:-1])
        density = DENS_MAP.get(dens_level)
        velocity = VPEN_MAP.get(v_level)
    except (ValueError, IndexError):
        print(f"  WARNING: could not parse {path.name}; skipping")
        return None

    if density is None:
        print(f"  WARNING: unknown density level {dens_level!r} in {path.name}; skipping")
        return None
    if velocity is None:
        print(f"  WARNING: unknown velocity level {v_level!r} in {path.name}; skipping")
        return None
    if s_value is None:
        print(f"  WARNING: missing scaling factor in {path.name}; skipping")
        return None

    return {
        "run_id": base,
        "dens_level": dens_level,
        "v_level": v_level,
        "S": s_value,
        "scenario_id": scenario_id,
        "ID": density,
        "v_pen": velocity,
    }


def match_on_depth(df: pd.DataFrame, depth_col: str = "z_m", val_col: str = "qb_MPa"):
    missing = [col for col in [depth_col, val_col] if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    df_sorted = (
        df[[depth_col, val_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(depth_col)
        .drop_duplicates(depth_col)
        .rename(columns={depth_col: "depth", val_col: "qb_MPa"})
        .reset_index(drop=True)
    )

    # Zero rows are pre-contact time steps, not physical resistance values.
    df_sorted = df_sorted[df_sorted["qb_MPa"] > 0].reset_index(drop=True)
    return df_sorted


def expected_csvs_from_metadata(metadata_path: Path):
    if not metadata_path.exists():
        return []

    expected = []
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            run_id = row.get("run_id", "")
            if run_id:
                expected.append(RAW_DIR / f"{run_id}.csv")
    return expected


def validate_expected_csvs(run, allow_partial: bool):
    expected = expected_csvs_from_metadata(run["metadata_path"])
    label = run["label"]

    if not expected:
        message = f"{label}: no metadata validation file found at {run['metadata_path']}"
        if allow_partial:
            print("WARNING: " + message)
            return
        raise SystemExit("ERROR: " + message)

    if len(expected) != EXPECTED_CSV_COUNT:
        message = (
            f"{label}: metadata lists {len(expected)} runs; expected "
            f"{EXPECTED_CSV_COUNT} for 4 x 3 x 5."
        )
        if allow_partial:
            print("WARNING: " + message)
        else:
            raise SystemExit("ERROR: " + message)

    missing = [path for path in expected if not path.exists()]
    if not missing:
        print(f"Metadata validation for {label}: all {len(expected)} expected CSV files found")
        return

    message = (
        f"{label}: missing {len(missing)} of {len(expected)} expected extracted CSV files. "
        "Run/postprocess the full matrix first, or rerun with --allow-partial."
    )
    if not allow_partial:
        raise SystemExit("ERROR: " + message)

    print("WARNING: " + message)
    for path in missing[:10]:
        print(f"  missing: {path.name}")
    if len(missing) > 10:
        print(f"  ... {len(missing) - 10} more missing file(s)")


def discover_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(RAW_DIR.glob(pattern))
    return sorted(set(files))


def build_dataset(run, allow_partial: bool = False):
    label = run["label"]
    out_dir = run["out_dir"]
    out_path = out_dir / "real_dataset.csv"
    plot_path = out_dir / "real_dataset_plot.csv"
    meta_path = out_dir / "dataset_meta.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    validate_expected_csvs(run, allow_partial)

    all_files = discover_files(run["patterns"])
    if not all_files:
        print(f"ERROR: no CSV files found in {RAW_DIR}")
        print(f"Patterns: {run['patterns']}")
        return None

    print("")
    print("=" * 60)
    print(f"Building dataset for {label}")
    print(f"RAW:    {RAW_DIR}")
    print(f"OUTPUT: {out_dir}")
    print(f"Found {len(all_files)} CSV files")

    file_info = []
    for path in all_files:
        info = parse_filename(path)
        if info is None:
            continue
        info["path"] = path
        file_info.append(info)
        print(
            f"  {path.name} -> DENS={info['dens_level']} ({info['ID']}), "
            f"V={info['v_level']} ({info['v_pen']}), S={info['S']}"
        )

    if not file_info:
        print(f"ERROR: no parseable CSV files found for {label}")
        return None

    df_info = pd.DataFrame(file_info)
    scenarios = df_info.groupby("scenario_id")
    print(f"\nFound {len(scenarios)} unique scenarios")

    records = []
    for scenario_id, grp in scenarios:
        ref_row = grp[grp["S"] == S_REF]
        if len(ref_row) == 0:
            print(f"  WARNING: no S={S_REF} reference for {scenario_id}; skipping")
            continue

        ref_path = ref_row.iloc[0]["path"]
        density = ref_row.iloc[0]["ID"]
        velocity = ref_row.iloc[0]["v_pen"]

        df_ref = pd.read_csv(ref_path)
        ref_grid = match_on_depth(df_ref).dropna()
        ref_grid["qb_MPa"] = smooth_curve(ref_grid["qb_MPa"].values, SMOOTH_WINDOW_QB)
        ref_grid["qb_grad"] = compute_depth_gradient(
            ref_grid["qb_MPa"].values, ref_grid["depth"].values
        )

        print(f"\nScenario: {scenario_id}")
        print(f"  ID={density}, v_pen={velocity}, ref rows={len(ref_grid)}")

        for _, fast_row in grp[grp["S"] != S_REF].iterrows():
            s_fast = fast_row["S"]
            fast_path = fast_row["path"]

            df_fast = pd.read_csv(fast_path)
            fast_grid = match_on_depth(df_fast).dropna()
            fast_grid["qb_MPa"] = smooth_curve(fast_grid["qb_MPa"].values, SMOOTH_WINDOW_QB)
            fast_grid["qb_grad"] = compute_depth_gradient(
                fast_grid["qb_MPa"].values, fast_grid["depth"].values
            )

            ref_rounded = ref_grid.copy()
            fast_rounded = fast_grid.copy()
            ref_rounded["depth"] = ref_rounded["depth"].round(3)
            fast_rounded["depth"] = fast_rounded["depth"].round(3)

            merged = pd.merge(
                ref_rounded.rename(columns={"qb_MPa": "qb_ref", "qb_grad": "qb_grad_ref"}),
                fast_rounded.rename(columns={"qb_MPa": "qb_fast", "qb_grad": "qb_fast_grad"}),
                on="depth",
            )
            print(f"  S={s_fast}: {len(merged)} common depth points")

            for _, row in merged.iterrows():
                records.append(
                    {
                        "soil_model": run["name"],
                        "scenario_id": scenario_id,
                        "run_id": fast_row["run_id"],
                        "ID": density,
                        "v_pen": velocity,
                        "S": s_fast,
                        "depth": round(float(row["depth"]), 6),
                        "qb_fast": round(float(row["qb_fast"]), 6),
                        "qb_fast_grad": round(float(row["qb_fast_grad"]), 6),
                        "qb_ref": round(float(row["qb_ref"]), 6),
                        "res_qb": round(float(row["qb_ref"] - row["qb_fast"]), 6),
                    }
                )

    if not records:
        print("\nERROR: no paired simulations found.")
        return None

    df_out = pd.DataFrame(records)
    n_before = len(df_out)

    print(f"\n  Outlier removal (threshold = {OUTLIER_STD_THRESH} sigma):")
    keep_mask = pd.Series(True, index=df_out.index)
    for s_value, grp in df_out.groupby("S"):
        mean = grp["res_qb"].mean()
        std = grp["res_qb"].std()
        if pd.isna(std) or std == 0:
            local_mask = pd.Series(True, index=grp.index)
        else:
            local_mask = (df_out.loc[grp.index, "res_qb"] - mean).abs() <= OUTLIER_STD_THRESH * std
        n_removed = int((~local_mask).sum())
        keep_mask.loc[grp.index] = local_mask
        print(f"    S={s_value}: removed {n_removed} rows (mean={mean:.4f}, std={std:.4f})")

    df_plot = df_out.copy()
    df_plot["is_shallow"] = (df_plot["depth"] < MIN_TRAINING_DEPTH).astype(int)
    df_plot.to_csv(plot_path, index=False)
    print(f"  Saved plot dataset -> {plot_path}")

    df_out = df_out[keep_mask].copy()
    n_after = len(df_out)
    print(f"  Total: {n_before} -> {n_after} rows ({n_before - n_after} removed)")

    df_out["is_shallow"] = (df_out["depth"] < MIN_TRAINING_DEPTH).astype(int)
    n_shallow = int(df_out["is_shallow"].sum())
    print(f"  Shallow zone (<{MIN_TRAINING_DEPTH}m): {n_shallow} rows flagged")

    qb_signal = df_out["qb_ref"].abs().replace(0, np.nan).mean()
    df_out["res_qb_norm"] = df_out["res_qb"] / qb_signal
    df_out.to_csv(out_path, index=False)

    meta = {
        "soil_model": run["name"],
        "soil_model_label": label,
        "raw_dir": str(RAW_DIR),
        "metadata_path": str(run["metadata_path"]),
        "expected_csv_count": EXPECTED_CSV_COUNT,
        "observed_csv_count": len(all_files),
        "S_ref": S_REF,
        "density_values": [0.3, 0.6, 0.8, 0.9],
        "velocity_values_cm_per_s": [25, 50, 100],
        "scaling_factors": [1, 10, 30, 50, 100],
        "smooth_window_qb": SMOOTH_WINDOW_QB,
        "smooth_window_qs": SMOOTH_WINDOW_QS,
        "min_training_depth": MIN_TRAINING_DEPTH,
        "outlier_std_thresh": OUTLIER_STD_THRESH,
        "gradient_window": GRADIENT_WINDOW,
        "qb_signal_full": round(float(qb_signal), 6),
        "note": (
            "qb_signal_full is for reference only. The training script adds "
            "qb_signal_train computed from training rows only."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Saved metadata -> {meta_path}")

    print("")
    print("=" * 60)
    print(f"Dataset built for {label}: {len(df_out):,} rows")
    print(f"Scenarios:     {df_out['scenario_id'].nunique()}")
    print(f"S values:      {sorted(df_out['S'].unique())}")
    print(f"ID values:     {sorted(df_out['ID'].unique())}")
    print(f"Saved ->       {out_path}")
    print("\nColumn summary:")
    print(df_out.describe().round(3).to_string())

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Build paired ML residual datasets from extracted per-run CSV files."
    )
    parser.add_argument(
        "--soil-model",
        choices=["all", "mcm", "mohr_coulomb", "hypoplastic"],
        default="all",
        help="'mohr_coulomb' is accepted as an alias for 'mcm'.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Build from available CSV files even when metadata validation is incomplete.",
    )
    args = parser.parse_args()

    selected = normalise_soil_model(args.soil_model)
    print(f"PROJECT: {PROJECT_ROOT}")
    print(f"DATA:    {DATA_ROOT}")
    print(f"RAW:     {RAW_DIR}")

    for run in SOIL_MODEL_RUNS:
        if selected != "all" and run["name"] != selected:
            continue
        build_dataset(run, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()

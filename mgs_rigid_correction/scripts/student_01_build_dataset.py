"""
01_build_dataset.py
-------------------
Reads all real Abaqus CSV files from the data folder.
Pairs fast runs (S=10, S=100) with the reference (S=1) 
using scenario_id as the matching key.


Output: data/real_dataset.csv

HOW TO USE:
  1. Put all 27 CSV files in the data/raw/ folder
  2. Run: python scripts/01_build_dataset.py

FILE NAMING CONVENTION EXPECTED:
  MC_G0_DENS_{DENS_LEVEL}_V_{V_LEVEL}_S{SSS}.csv
  Example: MC_G0_DENS_HIGH_V_HIGH_S001.csv

PARAMETER ENCODING IN FILENAME:
  DENS_HIGH  → density_level = HIGH
  DENS_MED   → density_level = MED
  DENS_LOW   → density_level = LOW  (if exists)
  V_HIGH     → velocity_level = HIGH
  V_MED      → velocity_level = MED  (if exists)
  V_REF      → velocity_level = REF  (reference velocity)
  S001       → S = 1   (reference — ground truth)
  S010       → S = 10
  S100       → S = 100 (fast run)
"""

import pandas as pd
import numpy as np
import os
import glob
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ──────────────────────────────────────────────────────────────────

_HERE   = os.path.dirname(os.path.abspath(__file__))

# Script should be in same folder as Data/
# Structure: first-Dataset/01_build_dataset.py + first-Dataset/Data/
def _find_root():
    for name in ["data", "Data", "DATA"]:
        if os.path.isdir(os.path.join(_HERE, name)):
            return _HERE, name
    for name in ["data", "Data", "DATA"]:
        candidate = os.path.join(_HERE, "..", name)
        if os.path.isdir(candidate):
            return os.path.normpath(os.path.join(_HERE, "..")), name
    return _HERE, "data"

_ROOT, _data_name = _find_root()
_DATA_FOLDER = os.path.join(_ROOT, _data_name)

RAW_DIR = os.path.join(_DATA_FOLDER, "extracted", "per_run_csv")
OUT_ROOT = os.path.join(_DATA_FOLDER, "processed", "student")
OUT_PATH = os.path.join(OUT_ROOT, "combined", "real_dataset.csv")
S_REF    = 1

SOIL_MODEL_RUNS = [
    {
        "name": "mcm",
        "label": "MCM",
        "patterns": ["MC_*.csv"],
        "out_path": os.path.join(OUT_ROOT, "mcm", "real_dataset.csv"),
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "patterns": ["G0_*.csv"],
        "out_path": os.path.join(OUT_ROOT, "hypoplastic", "real_dataset.csv"),
    },
]

print(f"ROOT:    {_ROOT}")
print(f"DATA:    {_DATA_FOLDER}")
print(f"RAW:     {RAW_DIR}")

# ── To run second batch (G0_ files) separately ───────────────────────────────
# RAW_DIR  = os.path.join(_DATA_FOLDER, "raw_g0")
# OUT_PATH = os.path.join(_DATA_FOLDER, "real_dataset_g0.csv")
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_GRID = np.arange(0.1, 9.05, 0.1)

DENS_MAP = {"MED": 0.60, "REF": 0.80, "HIGH": 0.90}
VPEN_MAP = {"LOW": 25.0, "REF": 50.0, "HIGH": 100.0}

# qb changes slowly with depth → larger window justified
# qs changes faster → keep smaller window
SMOOTH_WINDOW_QB = 30    # 30-point moving average
SMOOTH_WINDOW_QS = 30    # same window for both

# Shallow depth has extreme outliers and near-zero signal
# Model still predicts corrections for shallow at inference time
MIN_TRAINING_DEPTH = 1.0   # metres — rows below this depth excluded from training


# Remove rows where residual is beyond N standard deviations
# Removes interpolation artifacts and contact algorithm spikes
OUTLIER_STD_THRESHOLD = 3.0   # standard deviations

# ─── File parser ─────────────────────────────────────────────────────────────

def parse_filename(fname):
    """
    Extract parameters from filename.
    MC_G0_DENS_HIGH_V_HIGH_S001.csv
    Returns dict with: dens_level, v_level, S, scenario_id
    """
    base = os.path.basename(fname).replace(".csv", "")
    parts = base.split("_")

    # Find DENS level
    dens_idx = parts.index("DENS")
    dens_level = parts[dens_idx + 1]

    # Find V level
    v_idx = parts.index("V")
    v_level = parts[v_idx + 1]

    # Find S value (last part starts with S)
    s_part = [p for p in parts if p.startswith("S") and p[1:].isdigit()]
    S = int(s_part[0][1:]) if s_part else None

    # scenario_id = everything before the S part
    scenario_id = "_".join(parts[:-1])

    return {
        "dens_level": dens_level,
        "v_level":    v_level,
        "S":          S,
        "scenario_id": scenario_id,
        "ID":         DENS_MAP.get(dens_level, 0.80),
        "v_pen":      VPEN_MAP.get(v_level, 2.0),
    }


# ─── Smoothing ───────────────────────────────────────────────────────────────

def smooth_curve(series, window):
    """
    Apply centered moving average to smooth noisy simulation output.
    Applied only to S=1 reference curves.
    Uses separate windows for qb and qs based on their noise characteristics.
    """
    return pd.Series(series).rolling(
        window=window, center=True, min_periods=1
    ).mean().values


# ─── Interpolation ───────────────────────────────────────────────────────────

def interpolate_to_grid(df, depth_col="z_m", val_cols=["qb_MPa", "qs_kPa"],
                        grid=DEPTH_GRID):
    """
    Interpolate simulation output onto a common depth grid.
    This is necessary because different S values produce output
    at slightly different depth points due to different time steps.
    """
    df_sorted = df.sort_values(depth_col).drop_duplicates(depth_col)
    result = {"depth": grid}
    for col in val_cols:
        result[col] = np.interp(
            grid,
            df_sorted[depth_col].values,
            df_sorted[col].values,
            left=0.0,
            right=np.nan
        )
    return pd.DataFrame(result)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def build_dataset(label="Combined", patterns=None, out_path=OUT_PATH):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Find all CSV files — handles both MC_G0_ and G0_ naming conventions
    if patterns is None:
        patterns = ["MC_*.csv", "G0_*.csv"]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(RAW_DIR, pattern)))
    all_files = sorted(all_files)
    if not all_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        print(f"Patterns: {patterns}")
        return None

    print(f"\n{'='*50}")
    print(f"Building dataset for {label}")
    print(f"Output folder: {os.path.dirname(out_path)}")
    print(f"Found {len(all_files)} CSV files")

    # Parse all files and group by scenario_id
    file_info = []
    for fpath in all_files:
        info = parse_filename(fpath)
        info["path"] = fpath
        file_info.append(info)
        print(f"  {os.path.basename(fpath)} → "
              f"DENS={info['dens_level']} ({info['ID']}), "
              f"V={info['v_level']} ({info['v_pen']}), "
              f"S={info['S']}")

    df_info = pd.DataFrame(file_info)

    # Group by scenario_id (same setup, different S)
    scenarios = df_info.groupby("scenario_id")
    print(f"\nFound {len(scenarios)} unique scenarios")

    records = []
    for scenario_id, grp in scenarios:
        # Find reference file (S=S_REF)
        ref_row = grp[grp["S"] == S_REF]
        if len(ref_row) == 0:
            print(f"  WARNING: No S={S_REF} reference for {scenario_id} — skipping")
            continue

        ref_path = ref_row.iloc[0]["path"]
        ID       = ref_row.iloc[0]["ID"]
        v_pen    = ref_row.iloc[0]["v_pen"]

        # Read and interpolate reference
        df_ref   = pd.read_csv(ref_path)
        ref_grid = interpolate_to_grid(df_ref)
        ref_grid = ref_grid.dropna()

        # Improvement 1 — separate smoothing windows for qb and qs
        # qb: larger window (100) to reduce slow-changing noise
        ref_grid["qb_MPa"] = smooth_curve(ref_grid["qb_MPa"].values, SMOOTH_WINDOW_QB)
        ref_grid["qs_kPa"] = smooth_curve(ref_grid["qs_kPa"].values, SMOOTH_WINDOW_QS)
        print(f"  Smoothing applied: qb window={SMOOTH_WINDOW_QB}, qs window={SMOOTH_WINDOW_QS}")

        print(f"\nScenario: {scenario_id}")
        print(f"  ID={ID}, v_pen={v_pen}, ref rows={len(ref_grid)}")

        # For each fast run (S != S_REF)
        fast_rows = grp[grp["S"] != S_REF]
        for _, fast_row in fast_rows.iterrows():
            S_fast = fast_row["S"]
            fast_path = fast_row["path"]

            # Read and interpolate fast run
            df_fast   = pd.read_csv(fast_path)
            fast_grid = interpolate_to_grid(df_fast)
            fast_grid = fast_grid.dropna()

            # Apply same 30-point smoothing to fast runs
            # smoothing should be applied to all curves
            fast_grid["qb_MPa"] = smooth_curve(fast_grid["qb_MPa"].values, SMOOTH_WINDOW_QB)
            fast_grid["qs_kPa"] = smooth_curve(fast_grid["qs_kPa"].values, SMOOTH_WINDOW_QS)

            # Match on common depth points
            common_depths = np.intersect1d(
                np.round(ref_grid["depth"].values, 3),
                np.round(fast_grid["depth"].values, 3)
            )

            ref_matched  = ref_grid[np.round(ref_grid["depth"], 3).isin(
                np.round(common_depths, 3))]
            fast_matched = fast_grid[np.round(fast_grid["depth"], 3).isin(
                np.round(common_depths, 3))]

            print(f"  S={S_fast}: {len(common_depths)} common depth points")

            for i, depth in enumerate(common_depths):
                ref_qb  = ref_matched.iloc[i]["qb_MPa"]
                ref_qs  = ref_matched.iloc[i]["qs_kPa"]
                fast_qb = fast_matched.iloc[i]["qb_MPa"]
                fast_qs = fast_matched.iloc[i]["qs_kPa"]

                records.append({
                    "scenario_id": scenario_id,
                    "soil_model":  label,
                    "ID":          ID,
                    "v_pen":       v_pen,
                    "S":           S_fast,
                    "depth":       round(depth, 3),
                    "qb_fast":     round(fast_qb, 6),
                    "qs_fast":     round(fast_qs, 6),
                    "qb_ref":      round(ref_qb,  6),
                    "qs_ref":      round(ref_qs,  6),
                    "res_qb":      round(ref_qb  - fast_qb, 6),
                    "res_qs":      round(ref_qs  - fast_qs, 6),
                })

    if len(records) == 0:
        print("\nERROR: No paired simulations found.")
        print("This means no scenario has both a S=1 reference AND a fast run.")
        print("Make sure all 27 files are in the data/raw/ folder.")
        print("\nFiles found and their scenario_ids:")
        for _, row in df_info.iterrows():
            print(f"  S={row['S']:3d}  scenario={row['scenario_id']}")
        return None

    df_out = pd.DataFrame(records)
    n_before = len(df_out)

    # ── Improvement 5: normalise residuals by mean signal magnitude ───────────
    # Makes qb (MPa) and qs (kPa) comparable in scale for the model
    qb_signal = df_out["qb_ref"].abs().replace(0, np.nan).mean()
    qs_signal = df_out["qs_ref"].abs().replace(0, np.nan).mean()
    df_out["res_qb_norm"] = df_out["res_qb"] / qb_signal
    df_out["res_qs_norm"] = df_out["res_qs"] / qs_signal
    print(f"\n  Normalisation: qb signal mean={qb_signal:.3f} MPa, "
          f"qs signal mean={qs_signal:.3f} kPa")

    # ── Save full dataset BEFORE outlier removal — used for plotting only ────
    PLOT_PATH = out_path.replace(".csv", "_plot.csv")
    df_plot_save = df_out.copy()
    df_plot_save["is_shallow"] = (df_plot_save["depth"] < MIN_TRAINING_DEPTH).astype(int)
    df_plot_save.to_csv(PLOT_PATH, index=False)
    print(f"  Saved plot dataset (no outlier removal) → {PLOT_PATH}")

    # Removes interpolation artifacts and contact algorithm spikes
    # Uses raw (non-normalised) residuals for threshold computation
    for col in ["res_qb", "res_qs"]:
        mean = df_out[col].mean()
        std  = df_out[col].std()
        mask = (df_out[col] - mean).abs() <= OUTLIER_STD_THRESHOLD * std
        n_removed = (~mask).sum()
        df_out = df_out[mask]
        print(f"  Outlier removal {col}: removed {n_removed} rows "
              f"(>{OUTLIER_STD_THRESHOLD}σ)")

    # Shallow rows kept in dataset but flagged so training script can skip them
    # Model still uses full dataset for inference/evaluation
    df_out["is_shallow"] = (df_out["depth"] < MIN_TRAINING_DEPTH).astype(int)
    n_shallow = df_out["is_shallow"].sum()
    print(f"  Shallow zone (<{MIN_TRAINING_DEPTH}m): {n_shallow} rows flagged "
          f"(excluded from training, kept for evaluation)")

    n_after = len(df_out)
    print(f"  Total rows: {n_before} → {n_after} "
          f"({n_before - n_after} removed as outliers)")

    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*50}")
    print(f"Dataset built for {label}: {len(df_out):,} rows")
    print(f"Scenarios:     {df_out['scenario_id'].nunique()}")
    print(f"S values:      {sorted(df_out['S'].unique())}")
    print(f"ID values:     {sorted(df_out['ID'].unique())}")
    print(f"Saved →        {out_path}")
    print(f"\nColumn summary:")
    print(df_out.describe().round(3).to_string())

    return df_out


if __name__ == "__main__":
    for run in SOIL_MODEL_RUNS:
        build_dataset(
            label=run["label"],
            patterns=run["patterns"],
            out_path=run["out_path"],
        )

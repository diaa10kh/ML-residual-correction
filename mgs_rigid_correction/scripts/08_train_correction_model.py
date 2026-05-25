"""
Train residual correction models from the repo-adapted student dataset.

Input:
  data/processed/ml/{mcm,hypoplastic}/real_dataset.csv

Output:
  results/ml/{mcm,hypoplastic}/model_qb.pkl
  results/ml/{mcm,hypoplastic}/metrics.csv
  results/ml/{mcm,hypoplastic}/metrics_by_group.csv
  results/ml/{mcm,hypoplastic}/oof_predictions.csv
  results/ml/{mcm,hypoplastic}/test_predictions.csv
  results/ml/{mcm,hypoplastic}/validation_folds.csv
  plots/ml/{mcm,hypoplastic}/

Only high-S quantities are used as ML input features.  The S=1 response is
used only to form the target residual.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "ml"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ml"
PLOTS_ROOT = PROJECT_ROOT / "plots" / "ml"

RESULTS_DIR = RESULTS_ROOT / "combined"
PLOTS_DIR = PLOTS_ROOT / "combined"

TRAIN_MIN_DEPTH = 0.0

MODEL_PARAMS = dict(
    max_iter=600,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=42,
)

# One model is trained, then the predicted residual is scaled at application
# time.  This keeps the model conservative when small S values are already
# close to the S=1 reference.
S_DAMPING_ALPHA = {
    10: 1.0,
    30: 1.0,
    50: 1.0,
    100: 1.0,
}

SOIL_MODEL_RUNS = [
    {
        "name": "mcm",
        "label": "MCM",
        "data_path": DATA_ROOT / "mcm" / "real_dataset.csv",
        "meta_path": DATA_ROOT / "mcm" / "dataset_meta.json",
        "results_dir": RESULTS_ROOT / "mcm",
        "plots_dir": PLOTS_ROOT / "mcm",
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "data_path": DATA_ROOT / "hypoplastic" / "real_dataset.csv",
        "meta_path": DATA_ROOT / "hypoplastic" / "dataset_meta.json",
        "results_dir": RESULTS_ROOT / "hypoplastic",
        "plots_dir": PLOTS_ROOT / "hypoplastic",
    },
]


def normalise_soil_model(value: str) -> str:
    if value == "mohr_coulomb":
        return "mcm"
    if value == "both":
        return "all"
    return value


def feature_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build input features available from the high-MGS run only."""
    feat = pd.DataFrame(index=df.index)
    feat["ID"] = df["ID"]
    feat["v_pen"] = df["v_pen"]
    feat["S"] = df["S"]
    feat["depth"] = df["depth"]
    feat["qb_fast"] = df["qb_fast"]
    feat["D_m"] = feature_or_default(df, "D_m")
    feat["penetration_m"] = feature_or_default(df, "penetration_m")
    feat["penetration_over_D"] = feature_or_default(df, "penetration_over_D")
    feat["depth_over_D"] = feature_or_default(df, "depth_over_D")
    feat["depth_over_penetration"] = feature_or_default(df, "depth_over_penetration")

    if "qb_fast_grad" in df.columns:
        feat["qb_fast_grad"] = df["qb_fast_grad"].fillna(0.0)
    else:
        feat["qb_fast_grad"] = 0.0

    log_s = np.log10(df["S"].clip(lower=1))
    feat["S_x_depth"] = log_s * df["depth"]
    feat["ID_x_depth"] = df["ID"] * df["depth"]
    feat["S_x_ID"] = log_s * df["ID"]
    feat["v_x_S"] = df["v_pen"] * log_s
    feat["grad_x_S"] = feat["qb_fast_grad"] * log_s
    feat["grad_x_ID"] = feat["qb_fast_grad"] * df["ID"]
    feat["S_x_depth_over_D"] = log_s * feat["depth_over_D"]
    feat["S_x_penetration_over_D"] = log_s * feat["penetration_over_D"]

    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


FEATURE_NAMES = list(
    build_features(
        pd.DataFrame(
            {
                "ID": [0.9],
                "v_pen": [50.0],
                "S": [10],
                "depth": [5.0],
                "qb_fast": [10.0],
                "qb_fast_grad": [1.5],
                "D_m": [0.60],
                "penetration_m": [9.0],
                "penetration_over_D": [15.0],
                "depth_over_D": [8.333333],
                "depth_over_penetration": [0.555556],
            }
        )
    ).columns
)


def compute_metrics(y_true, y_pred, label: str = ""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "label": label,
        "WAPE(%)": wape_error(y_true, y_pred),
        "RMSE": rmse_error(y_true, y_pred),
        "n": len(y_true),
    }


def cross_validate(df: pd.DataFrame, target: str, n_splits=None):
    groups = df["scenario_id"].values
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print("  GroupKFold CV skipped: fewer than two scenarios")
        return np.nan, np.nan, []

    n_splits = min(n_splits or n_groups, n_groups)
    print(f"  GroupKFold CV: {n_splits} folds, {n_groups} unique scenarios")

    fold_rmses = []
    kf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df, groups=groups), start=1):
        df_tr = df.iloc[tr_idx]
        df_va = df.iloc[va_idx]
        x_tr = build_features(df_tr).values
        x_va = build_features(df_va).values
        y_tr = df_tr[target].values
        y_va = df_va[target].values
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(x_tr, y_tr)
        rmse = float(np.sqrt(mean_squared_error(y_va, model.predict(x_va))))
        fold_rmses.append(rmse)
        print(f"    fold {fold:02d}: RMSE={rmse:.5f}")

    mean_rmse = float(np.mean(fold_rmses))
    std_rmse = float(np.std(fold_rmses))
    print(f"  CV RMSE = {mean_rmse:.5f} +/- {std_rmse:.5f}")
    return mean_rmse, std_rmse, fold_rmses


def grouped_oof_predictions(df_train_pool: pd.DataFrame, full_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Create grouped out-of-fold predictions for every scenario.

    The validation rows are complete scenario curves from ``full_df``. The model
    for each fold is trained only on the remaining scenarios from
    ``df_train_pool``.
    """
    groups = df_train_pool["scenario_id"].values
    scenario_count = len(np.unique(groups))
    if scenario_count < 2:
        raise SystemExit("ERROR: need at least two scenarios for grouped validation")

    print(f"  Grouped out-of-fold validation: {scenario_count} folds over {scenario_count} scenarios")
    kf = GroupKFold(n_splits=scenario_count)
    parts = []
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(df_train_pool, groups=groups), start=1):
        df_tr = df_train_pool.iloc[tr_idx]
        scenario_ids = sorted(df_train_pool.iloc[va_idx]["scenario_id"].unique())
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(build_features(df_tr).values, df_tr[target].values)

        df_va = full_df[full_df["scenario_id"].isin(scenario_ids)].copy()
        df_va = apply_model_correction(df_va, model)
        df_va["validation_fold"] = fold
        parts.append(df_va)

        before = wape_error(df_va["qb_ref"], df_va["qb_fast"])
        after = wape_error(df_va["qb_ref"], df_va["qb_corrected"])
        rows.append(
            {
                "fold": fold,
                "scenario_id": ";".join(scenario_ids),
                "ID": df_va["ID"].iloc[0],
                "v_pen": df_va["v_pen"].iloc[0],
                "WAPE_before": before,
                "WAPE_after": after,
                "RMSE_before": rmse_error(df_va["qb_ref"], df_va["qb_fast"]),
                "RMSE_after": rmse_error(df_va["qb_ref"], df_va["qb_corrected"]),
                "n": len(df_va),
            }
        )
        print(f"    fold {fold:02d}: {', '.join(scenario_ids)} WAPE {before:.3f}% -> {after:.3f}%")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "validation_folds.csv", index=False)
    return pd.concat(parts, ignore_index=True)


def holdout_validation_summary(
    df_train_pool: pd.DataFrame,
    full_df: pd.DataFrame,
    target: str,
    group_col: str,
    strategy: str,
    output_name: str,
):
    """Run stricter holdout validation for one geometry-related grouping."""
    if group_col not in df_train_pool.columns or group_col not in full_df.columns:
        print(f"  {strategy} skipped: missing column {group_col}")
        return None

    train_groups = pd.Series(df_train_pool[group_col]).dropna().unique()
    if len(train_groups) < 2:
        print(f"  {strategy} skipped: fewer than two groups")
        return None

    print(f"  {strategy}: {len(train_groups)} folds by {group_col}")
    rows = []
    parts = []
    for fold, value in enumerate(sorted(train_groups, key=lambda item: str(item)), start=1):
        df_tr = df_train_pool[df_train_pool[group_col] != value]
        df_va = full_df[full_df[group_col] == value].copy()
        if df_tr.empty or df_va.empty:
            continue

        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(build_features(df_tr).values, df_tr[target].values)
        df_va = apply_model_correction(df_va, model)
        parts.append(df_va)

        before = wape_error(df_va["qb_ref"], df_va["qb_fast"])
        after = wape_error(df_va["qb_ref"], df_va["qb_corrected"])
        rows.append(
            {
                "strategy": strategy,
                "fold": fold,
                "holdout_column": group_col,
                "holdout_value": value,
                "train_rows": len(df_tr),
                "validation_rows": len(df_va),
                "WAPE_before": before,
                "WAPE_after": after,
                "RMSE_before": rmse_error(df_va["qb_ref"], df_va["qb_fast"]),
                "RMSE_after": rmse_error(df_va["qb_ref"], df_va["qb_corrected"]),
            }
        )
        print(f"    {group_col}={value}: WAPE {before:.3f}% -> {after:.3f}%")

    if not rows or not parts:
        print(f"  {strategy} skipped: no validation rows")
        return None

    pd.DataFrame(rows).to_csv(RESULTS_DIR / output_name, index=False)
    all_va = pd.concat(parts, ignore_index=True)
    return {
        "strategy": strategy,
        "folds": len(rows),
        "WAPE_before": wape_error(all_va["qb_ref"], all_va["qb_fast"]),
        "WAPE_after": wape_error(all_va["qb_ref"], all_va["qb_corrected"]),
        "RMSE_before": rmse_error(all_va["qb_ref"], all_va["qb_fast"]),
        "RMSE_after": rmse_error(all_va["qb_ref"], all_va["qb_corrected"]),
        "n": len(all_va),
    }


def save_robustness_validations(df_train_pool: pd.DataFrame, full_df: pd.DataFrame, df_oof: pd.DataFrame):
    summaries = [
        {
            "strategy": "grouped scenario OOF",
            "folds": int(df_oof["validation_fold"].nunique()) if "validation_fold" in df_oof.columns else np.nan,
            "WAPE_before": wape_error(df_oof["qb_ref"], df_oof["qb_fast"]),
            "WAPE_after": wape_error(df_oof["qb_ref"], df_oof["qb_corrected"]),
            "RMSE_before": rmse_error(df_oof["qb_ref"], df_oof["qb_fast"]),
            "RMSE_after": rmse_error(df_oof["qb_ref"], df_oof["qb_corrected"]),
            "n": len(df_oof),
        }
    ]
    for group_col, strategy, output_name in [
        ("geometry_id", "leave-one-geometry-out", "validation_leave_one_geometry.csv"),
        ("D_m", "leave-one-diameter-out", "validation_leave_one_diameter.csv"),
        ("penetration_m", "leave-one-penetration-out", "validation_leave_one_penetration.csv"),
    ]:
        summary = holdout_validation_summary(df_train_pool, full_df, "res_qb", group_col, strategy, output_name)
        if summary:
            summaries.append(summary)

    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(RESULTS_DIR / "validation_strategy_summary.csv", index=False)
    plot_validation_strategy_summary(df_summary)
    return df_summary


def train_final(df_train: pd.DataFrame, target: str, tag: str):
    x_train = build_features(df_train).values
    y_train = df_train[target].values
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(x_train, y_train)

    path = RESULTS_DIR / f"model_{tag}.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    print(f"  Saved model -> {path}")
    return model


def compute_and_save_normalisation(df_train: pd.DataFrame, meta_path: Path):
    qb_signal = df_train["qb_ref"].abs().replace(0, np.nan).mean()
    print(f"  Normalisation scalar from training rows only: {qb_signal:.4f} MPa")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["qb_signal_train"] = round(float(qb_signal), 6)
    meta["feature_names"] = FEATURE_NAMES
    meta["train_row_count"] = int(len(df_train))
    meta["train_scenario_count"] = int(df_train["scenario_id"].nunique())
    meta["train_min_depth_m"] = TRAIN_MIN_DEPTH
    meta["training_depth_policy"] = (
        "Full depth is used for training."
        if TRAIN_MIN_DEPTH <= 0
        else f"Rows with depth < {TRAIN_MIN_DEPTH} m are excluded from training."
    )
    meta["evaluation_depth_policy"] = (
        "Grouped out-of-fold validation evaluates complete scenario curves. "
        "The final saved model is trained on the full training pool."
    )
    meta["robustness_validation"] = (
        "Full-version datasets additionally write leave-one-geometry-out, "
        "leave-one-diameter-out, and leave-one-penetration-out summaries when "
        "the required geometry columns are available."
    )
    meta["correction_damping_alpha"] = S_DAMPING_ALPHA
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Updated metadata -> {meta_path}")
    return qb_signal


def corrected_frame(df: pd.DataFrame, model_qb) -> pd.DataFrame:
    out = df.copy()
    if "qb_corrected" not in out.columns:
        out = apply_model_correction(out, model_qb)
    elif "qb_residual_pred" not in out.columns:
        out["qb_residual_pred"] = out["qb_corrected"] - out["qb_fast"]
    if "correction_alpha" not in out.columns:
        out["correction_alpha"] = correction_alpha(out)
    return out


def correction_alpha(df: pd.DataFrame) -> pd.Series:
    return (
        df["S"]
        .round()
        .astype(int)
        .map(S_DAMPING_ALPHA)
        .fillna(1.0)
        .astype(float)
    )


def apply_model_correction(df: pd.DataFrame, model_qb) -> pd.DataFrame:
    out = df.copy()
    raw_pred = model_qb.predict(build_features(out).values) if len(out) else np.array([])
    alpha = correction_alpha(out)
    out["correction_alpha"] = alpha
    out["qb_residual_pred_raw"] = raw_pred
    out["qb_residual_pred"] = raw_pred * alpha.to_numpy()
    out["qb_corrected"] = (out["qb_fast"] + out["qb_residual_pred"]).clip(lower=0)
    return out


def corrected_visual_frame(df: pd.DataFrame, model_qb) -> pd.DataFrame:
    out = corrected_frame(df, model_qb)
    out["qb_corrected_plot"] = out["qb_corrected"]
    return out


def rmse_error(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if len(reference) == 0:
        return np.nan
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def wape_error(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    denominator = float(np.sum(np.abs(reference)))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(np.abs(reference - estimate)) / denominator * 100.0)


def compute_wape_group(df: pd.DataFrame, group_col: str, fast_col: str, corr_col: str, ref_col: str):
    rows = []
    for val, grp in df.groupby(group_col):
        rows.append(
            {
                group_col: val,
                "before": wape_error(grp[ref_col], grp[fast_col]),
                "after": wape_error(grp[ref_col], grp[corr_col]),
            }
        )
    return pd.DataFrame(rows).dropna()


def plot_number(value) -> str:
    value = float(value)
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def finite_vmax(values, default: float = 1.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return default
    return max(float(np.nanmax(finite)), default)


def validation_context(df: pd.DataFrame) -> str:
    if "validation_fold" in df.columns:
        return "Grouped out-of-fold validation"
    return "Final-model diagnostic"


def residual_damping_active() -> bool:
    return any(not np.isclose(alpha, 1.0) for alpha in S_DAMPING_ALPHA.values())


def plot_correction_curves(df_in: pd.DataFrame, model_qb, n: int = 20):
    df = corrected_visual_frame(df_in, model_qb)
    df["scenario_base"] = df["scenario_id"].str.replace("^MC_", "", regex=True)
    scenario_order = (
        df.groupby("scenario_base")
        .agg(ID=("ID", "first"), v_pen=("v_pen", "first"))
        .sort_values(["ID", "v_pen"])
    )

    scenarios = []
    for base in scenario_order.index.tolist():
        scenario_ids = df.loc[df["scenario_base"] == base, "scenario_id"].unique().tolist()
        mc_ids = [sid for sid in scenario_ids if str(sid).startswith("MC_")]
        scenarios.append(mc_ids[0] if mc_ids else scenario_ids[0])
    scenarios = scenarios[:n]
    if not scenarios:
        return

    ncols = 3
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        f"Correction curves - all {len(scenarios)} scenarios ({validation_context(df)})\n"
        "Base resistance qb [30-pt moving average applied in dataset]; green curve shows highest-S correction",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#f39c12", "#e74c3c", "#9b59b6", "#e67e22", "#3498db"]
    for i, scenario_id in enumerate(scenarios):
        ax = axes_flat[i]
        sub = df[df["scenario_id"] == scenario_id]
        s_values = sorted(int(s) for s in sub["S"].dropna().unique())
        if not s_values:
            ax.set_visible(False)
            continue

        for color, s_value in zip(colors, s_values):
            grp = sub[sub["S"] == s_value].sort_values("depth")
            ax.plot(grp["qb_fast"], grp["depth"], "--", color=color, lw=1.2, label=f"Fast (S={s_value})")

        ref_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        max_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        ax.plot(
            max_group["qb_corrected_plot"],
            max_group["depth"],
            color="#2ecc71",
            lw=2.5,
            label=f"ML Corrected (S={plot_number(s_values[-1])} only)",
        )
        ax.plot(
            ref_group["qb_ref"],
            ref_group["depth"],
            color="#2c3e50",
            lw=2.0,
            alpha=0.95,
            label="Reference (S=1)",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(f"ID={sub['ID'].iloc[0]}, v={sub['v_pen'].iloc[0]} cm/s")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(len(scenarios), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = PLOTS_DIR / "correction_curves_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_test_correction_curves(df_in: pd.DataFrame, model_qb):
    df = corrected_visual_frame(df_in, model_qb)
    scenario_order = (
        df.groupby("scenario_id")
        .agg(ID=("ID", "first"), v_pen=("v_pen", "first"))
        .sort_values(["ID", "v_pen"])
    )
    scenarios = scenario_order.index.tolist()
    if not scenarios:
        return

    ncols = 3
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        f"Correction curves - {len(scenarios)} grouped out-of-fold validation scenarios (Reference: S=1)\n"
        "Each scenario is predicted by a model trained without that scenario",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#f39c12", "#e74c3c", "#9b59b6", "#e67e22", "#3498db"]
    for i, scenario_id in enumerate(scenarios):
        ax = axes_flat[i]
        sub = df[df["scenario_id"] == scenario_id]
        s_values = sorted(int(s) for s in sub["S"].dropna().unique())
        if not s_values:
            ax.set_visible(False)
            continue

        for color, s_value in zip(colors, s_values):
            grp = sub[sub["S"] == s_value].sort_values("depth")
            ax.plot(grp["qb_fast"], grp["depth"], "--", color=color, lw=1.2, label=f"Fast (S={s_value})")

        ref_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        max_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        before = wape_error(max_group["qb_ref"], max_group["qb_fast"])
        after = wape_error(max_group["qb_ref"], max_group["qb_corrected"])
        ax.plot(
            max_group["qb_corrected_plot"],
            max_group["depth"],
            color="#2ecc71",
            lw=2.5,
            label="ML Corrected",
        )
        ax.plot(
            ref_group["qb_ref"],
            ref_group["depth"],
            color="#2c3e50",
            lw=2.0,
            alpha=0.95,
            label="Reference (S=1)",
        )
        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]")
        ax.set_ylabel("Depth [m]")
        ax.set_title(
            f"ID={sub['ID'].iloc[0]}, v={sub['v_pen'].iloc[0]} cm/s\n"
            f"S={s_values[-1]} WAPE {before:.1f}% -> {after:.1f}%",
            fontsize=10,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(len(scenarios), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    out = PLOTS_DIR / "correction_curves_test_scenarios.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_sensitivity(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    color_before = "#e74c3c"
    color_after = "#2ecc71"

    grouped = compute_wape_group(df, "S", "qb_fast", "qb_corrected", "qb_ref")
    if not grouped.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["S"], grouped["before"], "o--", color=color_before, lw=2, ms=7, label="Fast (raw)")
        ax.plot(grouped["S"], grouped["after"], "o-", color=color_after, lw=2, ms=7, label="After correction")
        ax.set_xlabel("Mass scaling factor S")
        ax.set_ylabel("WAPE (%) - Base resistance qb")
        ax.set_title(f"WAPE vs mass scaling factor S  ({validation_context(df)})", fontweight="bold")
        ax.set_xticks(grouped["S"].tolist())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / "sensitivity_vs_S.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

    grouped = compute_wape_group(df, "ID", "qb_fast", "qb_corrected", "qb_ref")
    if not grouped.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["ID"], grouped["before"], "s--", color=color_before, lw=2, ms=7, label="Fast (raw)")
        ax.plot(grouped["ID"], grouped["after"], "s-", color=color_after, lw=2, ms=7, label="After correction")
        ax.set_xlabel("Relative density ID")
        ax.set_ylabel("WAPE (%) - Base resistance qb")
        ax.set_title(f"WAPE vs relative density ID  ({validation_context(df)})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / "sensitivity_vs_ID.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

    s_values = sorted(df["S"].dropna().unique())
    id_values = sorted(df["ID"].dropna().unique())
    if s_values and id_values:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Heatmap: S x ID WAPE  ({validation_context(df)})", fontsize=12, fontweight="bold")
        for ax, (col_use, title, cmap) in zip(
            axes,
            [
                ("qb_fast", "Fast error WAPE% (before)", "Reds"),
                ("qb_corrected", "Corrected error WAPE% (after)", "Greens"),
            ],
        ):
            mat = np.full((len(id_values), len(s_values)), np.nan)
            for i, id_value in enumerate(id_values):
                for j, s_value in enumerate(s_values):
                    sub = df[(df["ID"] == id_value) & (df["S"] == s_value)]
                    mat[i, j] = wape_error(sub["qb_ref"], sub[col_use]) if not sub.empty else np.nan
            vmax = finite_vmax(mat)
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(s_values)))
            ax.set_xticklabels([plot_number(s) for s in s_values])
            ax.set_yticks(range(len(id_values)))
            ax.set_yticklabels([plot_number(i) for i in id_values])
            ax.set_xlabel("S")
            ax.set_ylabel("Density ID")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="WAPE (%)")
            for i in range(len(id_values)):
                for j in range(len(s_values)):
                    if np.isfinite(mat[i, j]):
                        color = "white" if mat[i, j] > vmax * 0.6 else "black"
                        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9, color=color)
        fig.tight_layout()
        out = PLOTS_DIR / "heatmap_S_vs_ID.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

    depth_max = float(df["depth"].max()) if len(df) else 0.0
    bins = np.arange(0, depth_max + 2, 2)
    if len(bins) >= 2 and s_values:
        labels = [f"{int(b)}-{int(b + 2)}m" for b in bins[:-1]]
        df_bins = df.copy()
        df_bins["depth_bin"] = pd.cut(df_bins["depth"], bins=bins, labels=labels, include_lowest=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Heatmap: S x depth WAPE  ({validation_context(df)})", fontsize=12, fontweight="bold")
        for ax, (col_use, title, cmap) in zip(
            axes,
            [
                ("qb_fast", "Fast error WAPE% by S and depth", "Reds"),
                ("qb_corrected", "Corrected error WAPE% by S and depth", "Greens"),
            ],
        ):
            mat = np.full((len(s_values), len(labels)), np.nan)
            for i, s_value in enumerate(s_values):
                for j, label in enumerate(labels):
                    sub = df_bins[(df_bins["S"] == s_value) & (df_bins["depth_bin"] == label)]
                    mat[i, j] = wape_error(sub["qb_ref"], sub[col_use]) if not sub.empty else np.nan
            vmax = finite_vmax(mat)
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(s_values)))
            ax.set_yticklabels([plot_number(s) for s in s_values])
            ax.set_xlabel("Depth bin")
            ax.set_ylabel("S")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="WAPE (%)")
            for i in range(len(s_values)):
                for j in range(len(labels)):
                    if np.isfinite(mat[i, j]):
                        color = "white" if mat[i, j] > vmax * 0.6 else "black"
                        ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color=color)
        fig.tight_layout()
        out = PLOTS_DIR / "heatmap_S_vs_depth.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")


def plot_publication_correction_curve(df_in: pd.DataFrame, model_qb):
    df = corrected_visual_frame(df_in, model_qb)
    for s_value in sorted(df["S"].dropna().unique()):
        df_s = df[df["S"] == s_value]
        best_id = None
        best_wape = np.inf
        for scenario_id, grp in df_s.groupby("scenario_id"):
            wape = wape_error(grp["qb_ref"], grp["qb_corrected"])
            if np.isfinite(wape) and wape < best_wape:
                best_id = scenario_id
                best_wape = wape
        if best_id is None:
            continue

        sub = df_s[df_s["scenario_id"] == best_id].sort_values("depth")
        id_value = sub["ID"].iloc[0]
        velocity = sub["v_pen"].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        before_wape = wape_error(sub["qb_ref"], sub["qb_fast"])
        fig.suptitle(
            f"Best-case Correction Curve - S={plot_number(s_value)}, "
            f"ID={plot_number(id_value)}, v={velocity:g} cm/s\n"
            f"{validation_context(df)}  |  WAPE {before_wape:.1f}% -> {best_wape:.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ax.plot(
            sub["qb_fast"],
            sub["depth"],
            "--",
            color="#e74c3c",
            lw=2.0,
            label=f"Fast (S={plot_number(s_value)})",
        )
        ax.plot(
            sub["qb_corrected_plot"],
            sub["depth"],
            "-",
            color="#2ecc71",
            lw=2.5,
            label="Corrected",
        )
        ax.plot(sub["qb_ref"], sub["depth"], "-", color="#2c3e50", lw=2.0, alpha=0.95, label="Reference (S=1)")
        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]", fontsize=11)
        ax.set_ylabel("Depth [m]", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = PLOTS_DIR / f"pub_A_correction_curve_S{plot_number(s_value)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")


def plot_error_reduction_summary(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    metric_specs = [
        ("WAPE (%)", wape_error),
        ("RMSE [MPa]", rmse_error),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Error Reduction Summary - Base resistance qb  ({validation_context(df)})",
        fontsize=13,
        fontweight="bold",
    )
    for ax, (metric_name, metric_fn) in zip(axes, metric_specs):
        before = []
        after = []
        for s_value in s_values:
            sub = df[df["S"] == s_value]
            before.append(metric_fn(sub["qb_ref"].values, sub["qb_fast"].values))
            after.append(metric_fn(sub["qb_ref"].values, sub["qb_corrected"].values))

        x = np.arange(len(s_values))
        width = 0.35
        bars_before = ax.bar(x - width / 2, before, width, color="#e74c3c", label="Before", alpha=0.85)
        bars_after = ax.bar(x + width / 2, after, width, color="#2ecc71", label="After", alpha=0.85)
        max_height = finite_vmax(before + after)
        for bar in list(bars_before) + list(bars_after):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max_height,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"S={plot_number(s)}" for s in s_values], fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "pub_B_error_reduction_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_residual_scatter(
    df_in: pd.DataFrame,
    model_qb,
    pred_col: str,
    ylabel: str,
    title: str,
    filename: str,
):
    df = corrected_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    fig, axes = plt.subplots(1, len(s_values), figsize=(7 * len(s_values), 6))
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for ax, s_value in zip(axes, s_values):
        sub = df[df["S"] == s_value]
        pred_res = sub[pred_col].values
        true_res = sub["res_qb"].values
        ax.scatter(true_res, pred_res, s=8, alpha=0.4, color="#3498db")
        lim = finite_vmax(np.concatenate([np.abs(true_res), np.abs(pred_res)])) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="Perfect (y=x)")
        ss_res = float(np.sum((true_res - pred_res) ** 2))
        ss_tot = float(np.sum((true_res - np.mean(true_res)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax.text(
            0.05,
            0.92,
            f"R2 = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.text(
            0.05,
            0.82,
            f"n scenarios = {sub['scenario_id'].nunique()}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Actual residual [MPa]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"S={plot_number(s_value)}")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_predicted_vs_actual(df_in: pd.DataFrame, model_qb):
    """Plot corrected qb directly against the S=1 reference qb."""
    df = corrected_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    fig, axes = plt.subplots(1, len(s_values), figsize=(7 * len(s_values), 6))
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        "Predicted vs Actual Base Resistance qb\n"
        f"{validation_context(df)}  |  Actual = S=1 reference",
        fontsize=12,
        fontweight="bold",
    )

    for ax, s_value in zip(axes, s_values):
        sub = df[df["S"] == s_value]
        ref = sub["qb_ref"].values
        fast = sub["qb_fast"].values
        corrected = sub["qb_corrected"].values

        limit = finite_vmax(np.concatenate([ref, fast, corrected])) * 1.03
        ax.scatter(ref, fast, s=7, alpha=0.18, color="#e74c3c", label="Fast raw")
        ax.scatter(ref, corrected, s=7, alpha=0.24, color="#2ecc71", label="Corrected")
        ax.plot([0, limit], [0, limit], "k--", lw=1.4, label="Perfect (y=x)")

        before_wape = wape_error(ref, fast)
        after_wape = wape_error(ref, corrected)
        before_rmse = rmse_error(ref, fast)
        after_rmse = rmse_error(ref, corrected)
        ax.text(
            0.05,
            0.92,
            f"WAPE {before_wape:.2f}% -> {after_wape:.2f}%\n"
            f"RMSE {before_rmse:.2f} -> {after_rmse:.2f} MPa",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Actual qb: reference S=1 [MPa]")
        ax.set_ylabel("Predicted qb [MPa]")
        ax.set_title(f"S={plot_number(s_value)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = PLOTS_DIR / "pub_C_predicted_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_residual_prediction_diagnostic(df_in: pd.DataFrame, model_qb):
    if not residual_damping_active():
        obsolete = PLOTS_DIR / "pub_C_raw_predicted_vs_actual.png"
        if obsolete.exists():
            obsolete.unlink()
        plot_residual_scatter(
            df_in,
            model_qb,
            pred_col="qb_residual_pred",
            ylabel="Predicted residual [MPa]",
            title=(
                "Predicted vs Actual Residual - Base resistance qb\n"
                f"{validation_context(df_in)}  |  No residual damping applied"
            ),
            filename="diagnostic_residual_predicted_vs_actual.png",
        )
        return

    plot_residual_scatter(
        df_in,
        model_qb,
        pred_col="qb_residual_pred_raw",
        ylabel="Raw predicted residual [MPa]",
        title=(
            "Raw Predicted vs Actual Residual - Base resistance qb\n"
            f"{validation_context(df_in)}  |  Before residual damping"
        ),
        filename="diagnostic_raw_residual_predicted_vs_actual.png",
    )
    plot_residual_scatter(
        df_in,
        model_qb,
        pred_col="qb_residual_pred",
        ylabel="Applied residual [MPa]",
        title=(
            "Applied vs Actual Residual - Base resistance qb\n"
            f"{validation_context(df_in)}  |  After residual damping"
        ),
        filename="diagnostic_residual_predicted_vs_actual.png",
    )


def plot_depth_error_profile(df_in: pd.DataFrame, model_qb):
    df = corrected_visual_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    fig, axes = plt.subplots(1, len(s_values), figsize=(7 * len(s_values), 7), sharey=True)
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        f"Error Profile vs Depth  ({validation_context(df)})\nMean +/- 1 std across all scenarios",
        fontsize=12,
        fontweight="bold",
    )
    smooth_window = 20

    for ax, s_value in zip(axes, s_values):
        sub_s = df[df["S"] == s_value]
        depths = sorted(sub_s["depth"].unique())
        fast_mean = []
        fast_std = []
        corr_mean = []
        corr_std = []
        for depth in depths:
            sub = sub_s[sub_s["depth"] == depth]
            fast_err = (sub["qb_fast"] - sub["qb_ref"]).values
            corr_err = (sub["qb_corrected_plot"] - sub["qb_ref"]).values
            fast_mean.append(np.mean(fast_err))
            fast_std.append(np.std(fast_err))
            corr_mean.append(np.mean(corr_err))
            corr_std.append(np.std(corr_err))

        depths_arr = np.asarray(depths)
        fast_mean = pd.Series(fast_mean).rolling(smooth_window, center=True, min_periods=1).mean().values
        corr_mean = pd.Series(corr_mean).rolling(smooth_window, center=True, min_periods=1).mean().values
        fast_std = pd.Series(fast_std).rolling(smooth_window, center=True, min_periods=1).mean().values
        corr_std = pd.Series(corr_std).rolling(smooth_window, center=True, min_periods=1).mean().values
        ax.plot(fast_mean, depths_arr, color="#e74c3c", lw=1.5, label="Fast error (mean)")
        ax.fill_betweenx(depths_arr, fast_mean - fast_std, fast_mean + fast_std, color="#e74c3c", alpha=0.15)
        ax.plot(corr_mean, depths_arr, color="#2ecc71", lw=2.0, label="Corrected error (mean)")
        ax.fill_betweenx(depths_arr, corr_mean - corr_std, corr_mean + corr_std, color="#2ecc71", alpha=0.15)
        ax.axvline(0, color="black", lw=1.2)
        ax.set_ylim(float(df["depth"].max()), 0.0)
        ax.set_xlabel("Error [MPa]")
        ax.set_title(f"S={plot_number(s_value)} - Base resistance qb")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Depth [m]")

    fig.tight_layout()
    out = PLOTS_DIR / "pub_D_depth_error_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_improvement_percentage(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    rows = []
    for (scenario_id, s_value), grp in df.groupby(["scenario_id", "S"]):
        before = wape_error(grp["qb_ref"], grp["qb_fast"])
        after = wape_error(grp["qb_ref"], grp["qb_corrected"])
        if np.isnan(before) or np.isnan(after):
            continue
        rows.append(
            {
                "scenario_id": scenario_id,
                "S": s_value,
                "ID": grp["ID"].iloc[0],
                "v_pen": grp["v_pen"].iloc[0],
                "improvement": before - after,
            }
        )
    if not rows:
        return

    df_imp = pd.DataFrame(rows).sort_values("improvement")
    fig, ax = plt.subplots(figsize=(10, max(5, len(df_imp) * 0.22)))
    colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in df_imp["improvement"]]
    labels = [
        f"S={plot_number(row.S)}, ID={plot_number(row.ID)}, v={plot_number(row.v_pen)}"
        for row in df_imp.itertuples()
    ]
    bars = ax.barh(np.arange(len(df_imp)), df_imp["improvement"], color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(df_imp)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=1.1)
    ax.set_xlabel("WAPE improvement (percentage points)")
    ax.set_title(
        f"Error Improvement per Scenario - Base resistance qb  ({validation_context(df)})\n"
        "Positive = correction helped, Negative = correction hurt",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    span = finite_vmax(df_imp["improvement"].abs().values)
    for bar, value in zip(bars, df_imp["improvement"]):
        offset = 0.03 * span if value >= 0 else -0.03 * span
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}pp",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    out = PLOTS_DIR / "pub_E_improvement_per_scenario.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_improvement_matrix(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    id_values = sorted(df["ID"].dropna().unique())
    v_values = sorted(df["v_pen"].dropna().unique())
    if not s_values or not id_values or not v_values:
        return

    rows = []
    for (s_value, id_value, v_value), grp in df.groupby(["S", "ID", "v_pen"]):
        before = wape_error(grp["qb_ref"], grp["qb_fast"])
        after = wape_error(grp["qb_ref"], grp["qb_corrected"])
        if np.isnan(before) or np.isnan(after):
            continue
        rows.append(
            {
                "S": s_value,
                "ID": id_value,
                "v_pen": v_value,
                "before": before,
                "after": after,
                "improvement": before - after,
            }
        )
    if not rows:
        return

    df_imp = pd.DataFrame(rows)
    span = finite_vmax(df_imp["improvement"].abs().values)
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, axes = plt.subplots(1, len(s_values), figsize=(5.1 * len(s_values), 4.3), sharey=True)
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        f"WAPE Improvement Matrix - Base resistance qb  ({validation_context(df)})",
        fontsize=13,
        fontweight="bold",
    )

    im = None
    for ax, s_value in zip(axes, s_values):
        mat = np.full((len(id_values), len(v_values)), np.nan)
        before_mat = np.full_like(mat, np.nan)
        after_mat = np.full_like(mat, np.nan)
        for i, id_value in enumerate(id_values):
            for j, v_value in enumerate(v_values):
                row = df_imp[
                    (df_imp["S"] == s_value)
                    & (df_imp["ID"] == id_value)
                    & (df_imp["v_pen"] == v_value)
                ]
                if row.empty:
                    continue
                mat[i, j] = row["improvement"].iloc[0]
                before_mat[i, j] = row["before"].iloc[0]
                after_mat[i, j] = row["after"].iloc[0]

        im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_title(f"S={plot_number(s_value)}")
        ax.set_xticks(range(len(v_values)))
        ax.set_xticklabels([plot_number(v) for v in v_values])
        ax.set_yticks(range(len(id_values)))
        ax.set_yticklabels([plot_number(i) for i in id_values])
        ax.set_xlabel("Penetration velocity [cm/s]")
        ax.grid(False)
        for i in range(len(id_values)):
            for j in range(len(v_values)):
                if not np.isfinite(mat[i, j]):
                    continue
                color = "white" if abs(mat[i, j]) > 0.55 * span else "black"
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:+.1f}pp\n{before_mat[i, j]:.1f}->{after_mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold" if mat[i, j] < 0 else "normal",
                )
    axes[0].set_ylabel("Relative density ID")
    fig.subplots_adjust(left=0.055, right=0.90, top=0.78, bottom=0.16, wspace=0.08)
    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.tolist(),
            shrink=0.82,
            pad=0.025,
            label="WAPE improvement (percentage points)",
        )
        cbar.ax.tick_params(labelsize=9)
    out = PLOTS_DIR / "pub_G_wape_improvement_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_validation_strategy_summary(df_summary: pd.DataFrame):
    if df_summary.empty:
        return

    labels = df_summary["strategy"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, df_summary["WAPE_before"], width, label="Raw fast", color="#d6604d")
    ax.bar(x + width / 2, df_summary["WAPE_after"], width, label="Corrected", color="#1b9e77")
    ax.set_ylabel("WAPE [%]")
    ax.set_title("Validation Strategy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for xpos, before, after in zip(x, df_summary["WAPE_before"], df_summary["WAPE_after"]):
        ax.text(
            xpos,
            max(before, after) * 1.02,
            f"{before:.1f}->{after:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    out = PLOTS_DIR / "validation_strategy_summary.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_correlation_matrix(df: pd.DataFrame, model_qb):
    feat = pd.DataFrame(index=df.index)
    feat["Density (ID)"] = df["ID"]
    feat["Velocity (v)"] = df["v_pen"]
    feat["Scaling (S)"] = df["S"]
    feat["Depth"] = df["depth"]
    feat["qb fast"] = df["qb_fast"]
    feat["qb fast grad"] = df["qb_fast_grad"] if "qb_fast_grad" in df.columns else 0.0
    for column, label in [
        ("D_m", "Diameter D"),
        ("penetration_m", "Penetration"),
        ("penetration_over_D", "Penetration/D"),
        ("depth_over_D", "Depth/D"),
        ("depth_over_penetration", "Depth/Penetration"),
    ]:
        if column in df.columns:
            feat[label] = pd.to_numeric(df[column], errors="coerce")
    log_s = np.log10(df["S"].clip(lower=1))
    feat["S x Depth"] = log_s * df["depth"]
    feat["ID x Depth"] = df["ID"] * df["depth"]
    feat["S x ID"] = log_s * df["ID"]
    feat["v x S"] = df["v_pen"] * log_s
    feat["Target (res_qb)"] = df["res_qb"]
    corr = feat.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.suptitle(
        "Correlation Matrix - Input Features and Target\n"
        "Last row/column shows correlation with res_qb (ML target)",
        fontsize=12,
        fontweight="bold",
    )
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlation coefficient")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    ncols = len(corr.columns)
    ax.add_patch(plt.Rectangle((-0.5, ncols - 1.5), ncols, 1, fill=False, edgecolor="#e74c3c", lw=2.5))
    ax.add_patch(plt.Rectangle((ncols - 1.5, -0.5), 1, ncols, fill=False, edgecolor="#e74c3c", lw=2.5))
    fig.tight_layout()
    out = PLOTS_DIR / "pub_F_correlation_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def save_metrics(df_test: pd.DataFrame):
    metrics = [
        compute_metrics(df_test["qb_ref"], df_test["qb_fast"], "qb fast vs ref"),
        compute_metrics(df_test["qb_ref"], df_test["qb_corrected"], "qb corrected vs ref"),
    ]

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + df_metrics.to_string(index=False))
    return df_metrics


def save_grouped_metrics(df_test: pd.DataFrame):
    rows = []
    for group_col in ["S", "ID", "v_pen", "scenario_id"]:
        for value, grp in df_test.groupby(group_col):
            rows.append(
                {
                    "group": group_col,
                    "value": value,
                    "WAPE_before": wape_error(grp["qb_ref"], grp["qb_fast"]),
                    "WAPE_after": wape_error(grp["qb_ref"], grp["qb_corrected"]),
                    "RMSE_before": rmse_error(grp["qb_ref"], grp["qb_fast"]),
                    "RMSE_after": rmse_error(grp["qb_ref"], grp["qb_corrected"]),
                    "n": len(grp),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "metrics_by_group.csv", index=False)
    return out


def train_soil_model(run):
    global RESULTS_DIR, PLOTS_DIR

    label = run["label"]
    data_path = run["data_path"]
    meta_path = run["meta_path"]
    RESULTS_DIR = run["results_dir"]
    PLOTS_DIR = run["plots_dir"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 60)
    print(f"MGS residual correction training - {label}")
    print("Reference: S=1")
    print("=" * 60)
    print(f"DATASET: {data_path}")
    print(f"RESULTS: {RESULTS_DIR}")
    print(f"PLOTS:   {PLOTS_DIR}")

    if not data_path.exists():
        raise SystemExit(f"ERROR: dataset not found for {label}: {data_path}")

    df = pd.read_csv(data_path).replace([np.inf, -np.inf], np.nan).dropna(
        subset=["scenario_id", "ID", "v_pen", "S", "depth", "qb_fast", "qb_ref", "res_qb"]
    )
    print(f"\nLoaded {len(df):,} rows")
    print(f"Scenarios: {df['scenario_id'].nunique()}")
    print(f"S values:  {sorted(df['S'].unique())}")
    print(f"ID values: {sorted(df['ID'].unique())}")
    print(f"Depth:     {df['depth'].min():.2f} - {df['depth'].max():.2f} m")

    if TRAIN_MIN_DEPTH > 0:
        df_train_pool = df[df["depth"] >= TRAIN_MIN_DEPTH].copy()
        print(
            f"Training depth policy: excluding rows with depth < {TRAIN_MIN_DEPTH:g} m. "
            "Evaluation still uses complete scenario curves."
        )
    else:
        df_train_pool = df.copy()
        print("Training depth policy: full depth included (TRAIN_MIN_DEPTH = 0.0 m)")

    if "is_shallow" in df.columns:
        print(
            f"Rows flagged shallow in dataset: {int((df['is_shallow'] == 1).sum())}. "
            "This flag is diagnostic unless TRAIN_MIN_DEPTH is set above 0."
        )
    print(f"Training pool: {len(df_train_pool):,} rows")

    if df_train_pool["scenario_id"].nunique() < 2:
        raise SystemExit(f"ERROR: need at least two scenarios to train {label}")

    print("\nTraining combined model for all S levels")
    cv_mean, cv_std, fold_rmses = cross_validate(df_train_pool, "res_qb")
    pd.DataFrame(
        [{"S": "all", "cv_rmse_mean": cv_mean, "cv_rmse_std": cv_std, "folds": len(fold_rmses)}]
    ).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

    df_oof = grouped_oof_predictions(df_train_pool, df, "res_qb")
    compute_and_save_normalisation(df_train_pool, meta_path)
    model_qb = train_final(df_train_pool, "res_qb", "qb")

    df_oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)
    # Backwards-compatible name used by older notebooks/docs.
    df_oof.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_metrics(df_oof)
    save_grouped_metrics(df_oof)
    save_robustness_validations(df_train_pool, df, df_oof)

    plot_path = data_path.with_name("real_dataset_plot.csv")
    if plot_path.exists():
        df_plot = pd.read_csv(plot_path)
        print(f"\nLoaded plot dataset: {len(df_plot):,} rows")
    else:
        df_plot = df.copy()
        print("\nPlot dataset not found; using training dataset for plots")

    for obsolete_plot in [
        PLOTS_DIR / "pub_A_representative_correction.png",
        PLOTS_DIR / "sensitivity_shallow_vs_deep.png",
        PLOTS_DIR / "pub_C_raw_predicted_vs_actual.png",
        PLOTS_DIR / "diagnostic_raw_residual_predicted_vs_actual.png",
    ]:
        if obsolete_plot.exists():
            obsolete_plot.unlink()

    print("\nGenerating plots...")
    plot_correction_curves(df_plot, model_qb)
    plot_test_correction_curves(df_oof, model_qb)
    plot_sensitivity(df_oof, model_qb)
    plot_publication_correction_curve(df_oof, model_qb)
    plot_error_reduction_summary(df_oof, model_qb)
    plot_predicted_vs_actual(df_oof, model_qb)
    plot_residual_prediction_diagnostic(df_oof, model_qb)
    plot_depth_error_profile(df_oof, model_qb)
    plot_improvement_percentage(df_oof, model_qb)
    plot_improvement_matrix(df_oof, model_qb)
    plot_correlation_matrix(df, model_qb)

    print(f"\nDone. Models -> {RESULTS_DIR}  Plots -> {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train residual correction models from paired ML datasets.")
    parser.add_argument(
        "--soil-model",
        choices=["all", "both", "mcm", "mohr_coulomb", "hypoplastic"],
        default="all",
        help="'mohr_coulomb' is accepted as an alias for 'mcm'.",
    )
    args = parser.parse_args()
    selected = normalise_soil_model(args.soil_model)

    for run in SOIL_MODEL_RUNS:
        if selected != "all" and run["name"] != selected:
            continue
        train_soil_model(run)


if __name__ == "__main__":
    main()

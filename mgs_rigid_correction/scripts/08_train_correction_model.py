"""
Train residual correction models from the repo-adapted student dataset.

Input:
  data/processed/ml/{mcm,hypoplastic}/real_dataset.csv

Output:
  results/ml/{mcm,hypoplastic}/model_qb.pkl
  results/ml/{mcm,hypoplastic}/metrics.csv
  results/ml/{mcm,hypoplastic}/test_predictions.csv
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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

SHALLOW_CUTOFF = 2.0
SEPARATE_S_MODELS = False

MODEL_PARAMS = dict(
    max_iter=600,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=10,
    l2_regularization=1.0,
    random_state=42,
)

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
    return value


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build input features available from the high-MGS run only."""
    feat = pd.DataFrame(index=df.index)
    feat["ID"] = df["ID"]
    feat["v_pen"] = df["v_pen"]
    feat["S"] = df["S"]
    feat["depth"] = df["depth"]
    feat["qb_fast"] = df["qb_fast"]

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
            }
        )
    ).columns
)


def compute_metrics(y_true, y_pred, label: str = ""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nrmse = float(rmse / (np.nanmax(y_true) - np.nanmin(y_true) + 1e-9))

    mask = np.abs(y_true) > 0.5
    if mask.sum() > 0:
        mpe = float(np.mean((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mpe = np.nan
        mape = np.nan

    return {
        "label": label,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE": nrmse,
        "MPE(%)": mpe,
        "MAPE(%)": mape,
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
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Updated metadata -> {meta_path}")
    return qb_signal


def corrected_frame(df: pd.DataFrame, model_qb) -> pd.DataFrame:
    out = df.copy()
    if "qb_corrected" not in out.columns:
        x = build_features(out).values
        out["qb_corrected"] = (out["qb_fast"] + model_qb.predict(x)).clip(lower=0)
    return out


def percentage_error(reference, estimate):
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    mask = np.abs(reference) > 0.5
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((reference[mask] - estimate[mask]) / reference[mask])) * 100)


def compute_mape_group(df: pd.DataFrame, group_col: str, fast_col: str, corr_col: str, ref_col: str):
    rows = []
    for val, grp in df.groupby(group_col):
        rows.append(
            {
                group_col: val,
                "before": percentage_error(grp[ref_col], grp[fast_col]),
                "after": percentage_error(grp[ref_col], grp[corr_col]),
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


def plot_correction_curves(df_in: pd.DataFrame, model_qb, n: int = 20):
    df = corrected_frame(df_in, model_qb)
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
        f"Correction curves - all {len(scenarios)} scenarios (Reference: S=1)\n"
        "Base resistance qb [30-pt moving average applied in dataset]",
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

        ref_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        ax.plot(
            ref_group["qb_ref"],
            ref_group["depth"],
            color="#2c3e50",
            lw=2.0,
            alpha=0.9,
            label="Reference (S=1)",
        )

        for color, s_value in zip(colors, s_values):
            grp = sub[sub["S"] == s_value].sort_values("depth")
            ax.plot(grp["qb_fast"], grp["depth"], "--", color=color, lw=1.2, label=f"Fast (S={s_value})")

        max_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        ax.plot(
            max_group["qb_corrected"],
            max_group["depth"],
            color="#2ecc71",
            lw=2.5,
            label="ML Corrected",
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


def plot_sensitivity(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    color_before = "#e74c3c"
    color_after = "#2ecc71"

    grouped = compute_mape_group(df, "S", "qb_fast", "qb_corrected", "qb_ref")
    if not grouped.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["S"], grouped["before"], "o--", color=color_before, lw=2, ms=7, label="Fast (raw)")
        ax.plot(grouped["S"], grouped["after"], "o-", color=color_after, lw=2, ms=7, label="After correction")
        ax.set_xlabel("Mass scaling factor S")
        ax.set_ylabel("MAPE (%) - Base resistance qb")
        ax.set_title("MAPE vs mass scaling factor S  (Reference: S=1)", fontweight="bold")
        ax.set_xticks(grouped["S"].tolist())
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / "sensitivity_vs_S.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

    grouped = compute_mape_group(df, "ID", "qb_fast", "qb_corrected", "qb_ref")
    if not grouped.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grouped["ID"], grouped["before"], "s--", color=color_before, lw=2, ms=7, label="Fast (raw)")
        ax.plot(grouped["ID"], grouped["after"], "s-", color=color_after, lw=2, ms=7, label="After correction")
        ax.set_xlabel("Relative density ID")
        ax.set_ylabel("MAPE (%) - Base resistance qb")
        ax.set_title("MAPE vs relative density ID  (Reference: S=1)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = PLOTS_DIR / "sensitivity_vs_ID.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out}")

    zones = [
        (f"Shallow (0-{SHALLOW_CUTOFF:g}m)", df["depth"] <= SHALLOW_CUTOFF),
        (f"Deep (>{SHALLOW_CUTOFF:g}m)", df["depth"] > SHALLOW_CUTOFF),
    ]
    before = [percentage_error(df.loc[mask, "qb_ref"], df.loc[mask, "qb_fast"]) for _, mask in zones]
    after = [percentage_error(df.loc[mask, "qb_ref"], df.loc[mask, "qb_corrected"]) for _, mask in zones]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(zones))
    ax.bar(x - 0.18, before, 0.35, color=color_before, label="Fast (raw)", alpha=0.85)
    ax.bar(x + 0.18, after, 0.35, color=color_after, label="After correction", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in zones])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Shallow vs Deep - Base resistance qb", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "sensitivity_shallow_vs_deep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")

    s_values = sorted(df["S"].dropna().unique())
    id_values = sorted(df["ID"].dropna().unique())
    if s_values and id_values:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Heatmap: S x ID MAPE  (Reference: S=1)", fontsize=12, fontweight="bold")
        for ax, (col_use, title, cmap) in zip(
            axes,
            [
                ("qb_fast", "Fast error MAPE% (before)", "Reds"),
                ("qb_corrected", "Corrected error MAPE% (after)", "Greens"),
            ],
        ):
            mat = np.full((len(id_values), len(s_values)), np.nan)
            for i, id_value in enumerate(id_values):
                for j, s_value in enumerate(s_values):
                    sub = df[(df["ID"] == id_value) & (df["S"] == s_value)]
                    mat[i, j] = percentage_error(sub["qb_ref"], sub[col_use]) if not sub.empty else np.nan
            vmax = finite_vmax(mat)
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(s_values)))
            ax.set_xticklabels([plot_number(s) for s in s_values])
            ax.set_yticks(range(len(id_values)))
            ax.set_yticklabels([plot_number(i) for i in id_values])
            ax.set_xlabel("S")
            ax.set_ylabel("Density ID")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="MAPE (%)")
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
        fig.suptitle("Heatmap: S x depth MAPE  (Reference: S=1)", fontsize=12, fontweight="bold")
        for ax, (col_use, title, cmap) in zip(
            axes,
            [
                ("qb_fast", "Fast error MAPE% by S and depth", "Reds"),
                ("qb_corrected", "Corrected error MAPE% by S and depth", "Greens"),
            ],
        ):
            mat = np.full((len(s_values), len(labels)), np.nan)
            for i, s_value in enumerate(s_values):
                for j, label in enumerate(labels):
                    sub = df_bins[(df_bins["S"] == s_value) & (df_bins["depth_bin"] == label)]
                    mat[i, j] = percentage_error(sub["qb_ref"], sub[col_use]) if not sub.empty else np.nan
            vmax = finite_vmax(mat)
            im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(s_values)))
            ax.set_yticklabels([plot_number(s) for s in s_values])
            ax.set_xlabel("Depth bin")
            ax.set_ylabel("S")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="MAPE (%)")
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
    df = corrected_frame(df_in, model_qb)
    for s_value in sorted(df["S"].dropna().unique()):
        df_s = df[df["S"] == s_value]
        best_id = None
        best_mape = np.inf
        for scenario_id, grp in df_s.groupby("scenario_id"):
            mape = percentage_error(grp["qb_ref"], grp["qb_corrected"])
            if np.isfinite(mape) and mape < best_mape:
                best_id = scenario_id
                best_mape = mape
        if best_id is None:
            continue

        sub = df_s[df_s["scenario_id"] == best_id].sort_values("depth")
        id_value = sub["ID"].iloc[0]
        velocity = sub["v_pen"].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(
            f"ML Correction Result - S={plot_number(s_value)}, ID={plot_number(id_value)}, v={velocity:g} cm/s\n"
            f"Reference: S=1  |  MAPE after correction = {best_mape:.1f}%",
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
        ax.plot(sub["qb_corrected"], sub["depth"], "-", color="#2ecc71", lw=2.5, label="Corrected")
        ax.plot(sub["qb_ref"], sub["depth"], "-", color="#2c3e50", lw=2.0, alpha=0.85, label="Reference (S=1)")
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
        ("RMSE [MPa]", lambda ref, pred: float(np.sqrt(np.mean((pred - ref) ** 2)))),
        ("MAE [MPa]", lambda ref, pred: float(np.mean(np.abs(pred - ref)))),
        ("MAPE (%)", percentage_error),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Error Reduction Summary - Base resistance qb  (Reference: S=1)", fontsize=13, fontweight="bold")
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


def plot_predicted_vs_actual(df_in: pd.DataFrame, model_qb):
    df = df_in.copy()
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    fig, axes = plt.subplots(1, len(s_values), figsize=(7 * len(s_values), 6))
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        "Predicted vs Actual Residual - Base resistance qb\n"
        "(Reference: S=1)  |  Points along diagonal = correct prediction",
        fontsize=11,
        fontweight="bold",
    )

    for ax, s_value in zip(axes, s_values):
        sub = df[df["S"] == s_value]
        pred_res = model_qb.predict(build_features(sub).values)
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
        ax.set_ylabel("Predicted residual [MPa]")
        ax.set_title(f"S={plot_number(s_value)}")
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = PLOTS_DIR / "pub_C_predicted_vs_actual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


def plot_depth_error_profile(df_in: pd.DataFrame, model_qb):
    df = corrected_frame(df_in, model_qb)
    s_values = sorted(df["S"].dropna().unique())
    if not s_values:
        return

    fig, axes = plt.subplots(1, len(s_values), figsize=(7 * len(s_values), 7), sharey=True)
    axes = np.ravel(axes) if hasattr(axes, "ravel") else np.asarray([axes])
    fig.suptitle(
        "Error Profile vs Depth  (Reference: S=1)\nMean +/- 1 std across all scenarios",
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
            corr_err = (sub["qb_corrected"] - sub["qb_ref"]).values
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
        ax.axhline(SHALLOW_CUTOFF, color="gray", lw=1, ls="--", alpha=0.6, label=f"Shallow cutoff {SHALLOW_CUTOFF:g}m")
        ax.invert_yaxis()
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
        before = percentage_error(grp["qb_ref"], grp["qb_fast"])
        after = percentage_error(grp["qb_ref"], grp["qb_corrected"])
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
    ax.set_xlabel("MAPE improvement (percentage points)")
    ax.set_title(
        "Error Improvement per Scenario - Base resistance qb  (Reference: S=1)\n"
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


def plot_correlation_matrix(df: pd.DataFrame, model_qb):
    feat = pd.DataFrame(index=df.index)
    feat["Density (ID)"] = df["ID"]
    feat["Velocity (v)"] = df["v_pen"]
    feat["Scaling (S)"] = df["S"]
    feat["Depth"] = df["depth"]
    feat["qb fast"] = df["qb_fast"]
    feat["qb fast grad"] = df["qb_fast_grad"] if "qb_fast_grad" in df.columns else 0.0
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
    metrics = []
    zones = [
        ("ALL depths", pd.Series(True, index=df_test.index)),
        (f"Shallow (0-{SHALLOW_CUTOFF:g}m)", df_test["depth"] <= SHALLOW_CUTOFF),
        (f"Deep (>{SHALLOW_CUTOFF:g}m)", df_test["depth"] > SHALLOW_CUTOFF),
    ]
    for zone_label, mask in zones:
        sub = df_test[mask]
        if sub.empty:
            continue
        metrics.append(compute_metrics(sub["qb_ref"], sub["qb_fast"], f"qb fast vs ref [{zone_label}]"))
        metrics.append(compute_metrics(sub["qb_ref"], sub["qb_corrected"], f"qb corrected vs ref [{zone_label}]"))

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + df_metrics.to_string(index=False))
    return df_metrics


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

    if "is_shallow" in df.columns:
        df_train_pool = df[df["is_shallow"] == 0].copy()
        print(f"Shallow rows excluded from training: {int((df['is_shallow'] == 1).sum())}")
    else:
        df_train_pool = df.copy()
    print(f"Training pool: {len(df_train_pool):,} rows")

    if df_train_pool["scenario_id"].nunique() < 2:
        raise SystemExit(f"ERROR: need at least two scenarios to train {label}")

    s_values = sorted(df_train_pool["S"].unique())
    if SEPARATE_S_MODELS and len(s_values) > 1:
        print(f"\nTraining separate models for each S level: {s_values}")
        models_qb = {}
        test_parts = []
        cv_rows = []
        train_parts = []
        for s_value in s_values:
            df_s = df_train_pool[df_train_pool["S"] == s_value]
            print(f"\n--- S={s_value}: {len(df_s):,} rows ---")
            cv_mean, cv_std, fold_rmses = cross_validate(df_s, "res_qb")
            cv_rows.append({"S": s_value, "cv_rmse_mean": cv_mean, "cv_rmse_std": cv_std, "folds": len(fold_rmses)})

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            tr_idx, te_idx = next(splitter.split(df_s, groups=df_s["scenario_id"]))
            df_tr = df_s.iloc[tr_idx]
            df_te = df_s.iloc[te_idx].copy()
            train_parts.append(df_tr)
            test_parts.append(df_te)
            models_qb[s_value] = train_final(df_tr, "res_qb", f"qb_S{s_value}")

        df_test = pd.concat(test_parts)
        df_test["qb_corrected"] = np.nan
        for s_value in s_values:
            mask = df_test["S"] == s_value
            x = build_features(df_test[mask]).values
            df_test.loc[mask, "qb_corrected"] = (
                df_test.loc[mask, "qb_fast"] + models_qb[s_value].predict(x)
            ).clip(lower=0)

        with (RESULTS_DIR / "models_qb_per_S.pkl").open("wb") as handle:
            pickle.dump(models_qb, handle)
        model_qb = models_qb[max(s_values)]
        with (RESULTS_DIR / "model_qb.pkl").open("wb") as handle:
            pickle.dump(model_qb, handle)
        compute_and_save_normalisation(pd.concat(train_parts), meta_path)
        pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    else:
        print("\nTraining combined model for all S levels")
        cv_mean, cv_std, fold_rmses = cross_validate(df_train_pool, "res_qb")
        pd.DataFrame(
            [{"S": "all", "cv_rmse_mean": cv_mean, "cv_rmse_std": cv_std, "folds": len(fold_rmses)}]
        ).to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(splitter.split(df_train_pool, groups=df_train_pool["scenario_id"]))
        df_train = df_train_pool.iloc[tr_idx]
        df_test = df_train_pool.iloc[te_idx].copy()

        print(
            f"Train: {df_train['scenario_id'].nunique()} scenarios; "
            f"Test: {df_test['scenario_id'].nunique()} scenarios"
        )
        compute_and_save_normalisation(df_train, meta_path)
        model_qb = train_final(df_train, "res_qb", "qb")
        df_test["qb_corrected"] = (
            df_test["qb_fast"] + model_qb.predict(build_features(df_test).values)
        ).clip(lower=0)

    df_test.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    save_metrics(df_test)

    plot_path = data_path.with_name("real_dataset_plot.csv")
    if plot_path.exists():
        df_plot = pd.read_csv(plot_path)
        print(f"\nLoaded plot dataset: {len(df_plot):,} rows")
    else:
        df_plot = df.copy()
        print("\nPlot dataset not found; using training dataset for plots")

    obsolete_plot = PLOTS_DIR / "pub_A_representative_correction.png"
    if obsolete_plot.exists():
        obsolete_plot.unlink()

    print("\nGenerating plots...")
    plot_correction_curves(df_plot, model_qb)
    plot_sensitivity(df_plot, model_qb)
    plot_publication_correction_curve(df_plot, model_qb)
    plot_error_reduction_summary(df_test, model_qb)
    plot_predicted_vs_actual(df_plot, model_qb)
    plot_depth_error_profile(df_plot, model_qb)
    plot_improvement_percentage(df_test, model_qb)
    plot_correlation_matrix(df, model_qb)

    print(f"\nDone. Models -> {RESULTS_DIR}  Plots -> {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Train residual correction models from paired ML datasets.")
    parser.add_argument(
        "--soil-model",
        choices=["all", "mcm", "mohr_coulomb", "hypoplastic"],
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

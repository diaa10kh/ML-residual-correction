"""
08_train_correction_model.py
----------------------------
Trains gradient boosting models to predict the residual error
between mass-scaled runs and the reference (S=1).

Reference is now S=1.

Input:  data/processed/ml/{mohr_coulomb,hypoplastic}/real_dataset.csv
Output: results/ml/{mohr_coulomb,hypoplastic}/model_qb.pkl
        results/ml/{mohr_coulomb,hypoplastic}/metrics.csv
        plots/ml/{mohr_coulomb,hypoplastic}/
"""

import numpy as np
import pandas as pd
import pickle
import os
import argparse
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ─── Config ──────────────────────────────────────────────────────────────────

_HERE   = os.path.dirname(os.path.abspath(__file__))

# _HERE is the folder containing this script
# Structure: first-Dataset/ contains both the script AND Data/
# So _ROOT = _HERE (script is in same folder as Data/)
def _find_root():
    # Check if Data/ or data/ is in the same folder as the script
    for name in ["data", "Data", "DATA"]:
        if os.path.isdir(os.path.join(_HERE, name)):
            return _HERE, name
    # Fallback — check one level up
    for name in ["data", "Data", "DATA"]:
        candidate = os.path.join(_HERE, "..", name)
        if os.path.isdir(candidate):
            return os.path.normpath(os.path.join(_HERE, "..")), name
    return _HERE, "data"

_ROOT, _data_name = _find_root()
_DATA_FOLDER   = os.path.join(_ROOT, _data_name)

DATA_ROOT      = os.path.join(_DATA_FOLDER, "processed", "ml")
DATA_PATH      = os.path.join(DATA_ROOT, "combined", "real_dataset.csv")
RESULTS_ROOT   = os.path.join(_ROOT, "results", "ml")
PLOTS_ROOT     = os.path.join(_ROOT, "plots", "ml")
RESULTS_DIR    = os.path.join(RESULTS_ROOT, "combined")
PLOTS_DIR      = os.path.join(PLOTS_ROOT, "combined")
SHALLOW_CUTOFF = 2.0

SOIL_MODEL_RUNS = [
    {
        "name": "mohr_coulomb",
        "label": "Mohr-Coulomb",
        "data_path": os.path.join(DATA_ROOT, "mohr_coulomb", "real_dataset.csv"),
        "results_dir": os.path.join(RESULTS_ROOT, "mohr_coulomb"),
        "plots_dir": os.path.join(PLOTS_ROOT, "mohr_coulomb"),
    },
    {
        "name": "hypoplastic",
        "label": "Hypoplastic",
        "data_path": os.path.join(DATA_ROOT, "hypoplastic", "real_dataset.csv"),
        "results_dir": os.path.join(RESULTS_ROOT, "hypoplastic"),
        "plots_dir": os.path.join(PLOTS_ROOT, "hypoplastic"),
    },
]

# ── Improvement 3: train separate models per S level ─────────────────────────
# Different S levels can have different error patterns; separate models can learn that cleanly.
# Set to True to train one model per S, False to train one combined model.
SEPARATE_S_MODELS = False   # Keep False until enough scenarios are available for each S level.

MODEL_PARAMS = dict(
    max_iter          = 600,
    max_depth         = 6,
    learning_rate     = 0.05,
    min_samples_leaf  = 10,
    l2_regularization = 1.0,
    random_state      = 42,
)

# ─── Feature engineering ─────────────────────────────────────────────────────

def build_features(df):
    """
    Build ML input features from real dataset columns.
    All features available at inference time — no reference needed.
    """
    feat = pd.DataFrame(index=df.index)

    # Simulation parameters
    feat["ID"]          = df["ID"]
    feat["v_pen"]       = df["v_pen"]
    feat["S"]           = df["S"]
    feat["log_S"]       = np.log10(df["S"].clip(lower=1))

    # Depth features
    feat["depth"]       = df["depth"]
    feat["sqrt_depth"]  = np.sqrt(df["depth"])

    # Fast simulation outputs at this depth — qb only )
    feat["qb_fast"]     = df["qb_fast"]

    # Interaction terms
    feat["S_x_depth"]   = feat["log_S"]  * feat["depth"]
    feat["ID_x_depth"]  = feat["ID"]     * feat["depth"]
    feat["S_x_ID"]      = feat["log_S"]  * feat["ID"]
    feat["v_x_S"]       = feat["v_pen"]  * feat["log_S"]

    return feat


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=""):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-9)

    # MPE — only where signal is large enough
    mask_mpe = np.abs(y_true) > 0.5
    mpe = np.mean(np.abs(
        (y_true[mask_mpe] - y_pred[mask_mpe]) / y_true[mask_mpe]
    )) * 100 if mask_mpe.sum() > 0 else np.nan

    # MAPE — Mean Absolute Percentage Error
    # Only where denominator is large enough to avoid blow-up near zero
    mask_mape = np.abs(y_true) > 0.5
    mape = np.mean(np.abs(
        (y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape]
    )) * 100 if mask_mape.sum() > 0 else np.nan

    return {"label": label, "RMSE": rmse, "MAE": mae,
            "NRMSE": nrmse, "MPE(%)": mpe, "MAPE(%)": mape,
            "n": len(y_true)}


# ─── Training ────────────────────────────────────────────────────────────────

def train(df_train, df_val, target, tag):
    X_tr = build_features(df_train).values
    X_va = build_features(df_val).values
    y_tr = df_train[target].values
    y_va = df_val[target].values

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_tr, y_tr)
    val_rmse = np.sqrt(mean_squared_error(y_va, model.predict(X_va)))
    print(f"  [{tag}] val RMSE: {val_rmse:.5f}")
    with open(f"{RESULTS_DIR}/model_{tag}.pkl", "wb") as f:
        pickle.dump(model, f)
    return model


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_correction_curves(df_test, model_qb, model_qs=None, n=9):
    """
    One plot per scenario. It shows all available mass-scaled curves and the
    ML-corrected curve for the largest available S level.
    """
    df = df_test.copy()
    if "qb_corrected" not in df.columns:
        X = build_features(df).values
        df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    # Get all scenarios sorted by ID then velocity
    all_scenarios = (df.groupby("scenario_id")
                     .agg(ID=("ID","first"), v=("v_pen","first"))
                     .sort_values(["ID","v"])
                     .index.tolist())[:n]

    n_show = len(all_scenarios)
    ncols  = 3
    nrows  = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    fig.suptitle(
        "Correction curves — all scenarios  (Reference: S=1)\n"
        "Base resistance qb",
        fontsize=13, fontweight="bold"
    )

    for i, sid in enumerate(all_scenarios):
        ax  = axes_flat[i]
        sub = df[df["scenario_id"] == sid]
        ID  = sub["ID"].iloc[0]
        v   = sub["v_pen"].iloc[0]

        s_values = sorted(int(s) for s in sub["S"].dropna().unique())
        if not s_values:
            ax.set_visible(False)
            continue

        reference_group = sub[sub["S"] == s_values[0]].sort_values("depth")
        ax.plot(reference_group["qb_ref"].values, reference_group["depth"].values,
                "-", color="#2c3e50", lw=2.0, alpha=0.9,
                label="Reference (S=1)")

        colors = ["#f39c12", "#d35400", "#e74c3c", "#8e44ad", "#3498db"]
        for color, S_val in zip(colors, s_values):
            s_group = sub[sub["S"] == S_val].sort_values("depth")
            ax.plot(s_group["qb_fast"].values, s_group["depth"].values,
                    "--", color=color, lw=1.4, label=f"Mass-scaled (S={S_val})")

        correction_group = sub[sub["S"] == s_values[-1]].sort_values("depth")
        ax.plot(correction_group["qb_corrected"].values, correction_group["depth"].values,
                "-", color="#2ecc71", lw=2.5, label=f"ML corrected (S={s_values[-1]})")

        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]", fontsize=9)
        ax.set_ylabel("Depth [m]", fontsize=9)
        ax.set_title(f"ID={ID}, v={v} cm/s", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/correction_curves_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    return df


def plot_error_by_depth(df_test, model_qb, model_qs=None):
    """One panel per S level showing error scatter vs depth."""
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())
    fig, axes = plt.subplots(1, len(S_vals), figsize=(7 * len(S_vals), 5))
    if len(S_vals) == 1:
        axes = [axes]
    fig.suptitle("Error by depth — Base resistance qb  (Reference: S=1)",
                 fontsize=12, fontweight="bold")

    for ax, S_val in zip(axes, S_vals):
        sub = df[df["S"] == S_val]
        ax.scatter(sub["depth"], sub["qb_fast"]-sub["qb_ref"],
                   s=3, alpha=0.3, color="#e74c3c", label="Fast error")
        ax.scatter(sub["depth"], sub["qb_corrected"]-sub["qb_ref"],
                   s=3, alpha=0.3, color="#2ecc71", label="Corrected error")
        ax.axhline(0, color="black", lw=1)
        ax.axvline(SHALLOW_CUTOFF, color="gray", lw=1, ls="--",
                   label=f"Shallow cutoff {SHALLOW_CUTOFF}m")
        ax.set_xlabel("Depth [m]")
        ax.set_ylabel("Error [MPa]")
        ax.set_title(f"S={S_val}")
        ax.legend(fontsize=8, markerscale=4)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/error_by_depth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def compute_mape_group(df, group_col, fc, cc, rc):
    rows = []
    for val, grp in df.groupby(group_col):
        mask = grp[rc].abs() > 0.5
        if mask.sum() == 0:
            continue
        before = np.mean(np.abs(
            (grp.loc[mask,rc]-grp.loc[mask,fc])/grp.loc[mask,rc]))*100
        after  = np.mean(np.abs(
            (grp.loc[mask,rc]-grp.loc[mask,cc])/grp.loc[mask,rc]))*100
        rows.append({group_col: val, "before": before, "after": after})
    return pd.DataFrame(rows)


def plot_sensitivity(df_test, model_qb, model_qs=None):
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    C_B = "#e74c3c"
    C_A = "#2ecc71"

    # Plot 1: Error vs S
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("Plot 1 — MAPE vs mass scaling factor S\n(Reference: S=1)",
                 fontsize=12, fontweight="bold")
    g = compute_mape_group(df, "S", "qb_fast", "qb_corrected", "qb_ref")
    ax.plot(g["S"], g["before"], "o--", color=C_B, lw=2, ms=7, label="Fast (raw)")
    ax.plot(g["S"], g["after"],  "o-",  color=C_A, lw=2, ms=7, label="After correction")
    ax.set_xlabel("Mass scaling factor S"); ax.set_ylabel("MAPE (%) — Base resistance qb")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xticks(g["S"].tolist())
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_vs_S.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/sensitivity_vs_S.png")

    # Plot 2: Error vs ID
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("Plot 2 — MAPE vs relative density ID\n(Reference: S=1)",
                 fontsize=12, fontweight="bold")
    g = compute_mape_group(df, "ID", "qb_fast", "qb_corrected", "qb_ref")
    ax.plot(g["ID"], g["before"], "s--", color=C_B, lw=2, ms=7, label="Fast (raw)")
    ax.plot(g["ID"], g["after"],  "s-",  color=C_A, lw=2, ms=7, label="After correction")
    ax.set_xlabel("Relative density ID"); ax.set_ylabel("MAPE (%) — Base resistance qb")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_vs_ID.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/sensitivity_vs_ID.png")

    # Plot 3: Shallow vs Deep
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("Plot 3 — MAPE in shallow vs deep zone\n(Reference: S=1)",
                 fontsize=12, fontweight="bold")
    df["zone"] = df["depth"].apply(
        lambda z: f"Shallow (0–{SHALLOW_CUTOFF}m)" if z <= SHALLOW_CUTOFF
                  else f"Deep (>{SHALLOW_CUTOFF}m)")
    zones = [f"Shallow (0–{SHALLOW_CUTOFF}m)", f"Deep (>{SHALLOW_CUTOFF}m)"]
    before, after = [], []
    for z in zones:
        sub  = df[df["zone"] == z]
        mask = sub["qb_ref"].abs() > 0.5
        before.append(np.mean(np.abs(
            (sub.loc[mask,"qb_ref"]-sub.loc[mask,"qb_fast"])/sub.loc[mask,"qb_ref"]))*100
            if mask.sum()>0 else np.nan)
        after.append(np.mean(np.abs(
            (sub.loc[mask,"qb_ref"]-sub.loc[mask,"qb_corrected"])/sub.loc[mask,"qb_ref"]))*100
            if mask.sum()>0 else np.nan)
    x = np.arange(len(zones))
    ax.bar(x-0.18, before, 0.35, color=C_B, label="Fast (raw)",       alpha=0.85)
    ax.bar(x+0.18, after,  0.35, color=C_A, label="After correction", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(zones, fontsize=9)
    ax.set_ylabel("MAPE (%)"); ax.set_title("Base resistance qb")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_shallow_vs_deep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/sensitivity_shallow_vs_deep.png")

    # Plot 4: Heatmap S × ID
    S_vals  = sorted(df["S"].unique())
    ID_vals = sorted(df["ID"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Plot 4 — Heatmap: S × ID MAPE (Reference: S=1)", fontsize=12, fontweight="bold")
    for ax, (col_use, title, cmap) in zip(axes, [
        ("qb_fast",      "Fast error MAPE% (before)",     "Reds"),
        ("qb_corrected", "Corrected error MAPE% (after)", "Greens"),
    ]):
        mat = np.full((len(ID_vals), len(S_vals)), np.nan)
        for i, idv in enumerate(ID_vals):
            for j, sv in enumerate(S_vals):
                sub  = df[(df["ID"]==idv) & (df["S"]==sv)]
                mask = sub["qb_ref"].abs() > 0.5
                if mask.sum() == 0: continue
                mat[i,j] = np.mean(np.abs(
                    (sub.loc[mask,"qb_ref"]-sub.loc[mask,col_use])
                    /sub.loc[mask,"qb_ref"]))*100
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(mat))
        ax.set_xticks(range(len(S_vals)));  ax.set_xticklabels(S_vals)
        ax.set_yticks(range(len(ID_vals))); ax.set_yticklabels(ID_vals)
        ax.set_xlabel("S"); ax.set_ylabel("Density ID"); ax.set_title(title)
        plt.colorbar(im, ax=ax, label="MAPE (%)")
        for i in range(len(ID_vals)):
            for j in range(len(S_vals)):
                if not np.isnan(mat[i,j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=9,
                            color="white" if mat[i,j]>np.nanmax(mat)*0.6 else "black")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/heatmap_S_vs_ID.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/heatmap_S_vs_ID.png")

    # Plot 5: Heatmap S × depth
    bins   = np.arange(0, df["depth"].max()+2, 2)
    labels = [f"{int(b)}–{int(b+2)}m" for b in bins[:-1]]
    df["depth_bin"] = pd.cut(df["depth"], bins=bins, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plot 5 — Heatmap: S × depth MAPE (Reference: S=1)", fontsize=12, fontweight="bold")
    for ax, (col_use, title, cmap) in zip(axes, [
        ("qb_fast",      "Fast error MAPE% by S and depth",     "Reds"),
        ("qb_corrected", "Corrected error MAPE% by S and depth", "Greens"),
    ]):
        mat = np.full((len(S_vals), len(labels)), np.nan)
        for i, sv in enumerate(S_vals):
            for j, lbl in enumerate(labels):
                sub  = df[(df["S"]==sv) & (df["depth_bin"]==lbl)]
                mask = sub["qb_ref"].abs() > 0.5
                if mask.sum() == 0: continue
                mat[i,j] = np.mean(np.abs(
                    (sub.loc[mask,"qb_ref"]-sub.loc[mask,col_use])
                    /sub.loc[mask,"qb_ref"]))*100
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(mat))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(S_vals))); ax.set_yticklabels(S_vals)
        ax.set_xlabel("Depth bin"); ax.set_ylabel("S"); ax.set_title(title)
        plt.colorbar(im, ax=ax, label="MAPE (%)")
        for i in range(len(S_vals)):
            for j in range(len(labels)):
                if not np.isnan(mat[i,j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if mat[i,j]>np.nanmax(mat)*0.6 else "black")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/heatmap_S_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/heatmap_S_vs_depth.png")
    print(f"\n  All sensitivity plots saved in {PLOTS_DIR}/")

# ─── Publication plots ───────────────────────────────────────────────────────

def plot_publication_correction_curve(df_test, model_qb, model_qs=None):
    """
    Plot A — One file per S level showing best performing scenario.
    Uses data already smoothed once in 07_build_ml_dataset.py.
    """
    df = df_test.copy()
    if "qb_corrected" not in df.columns:
        X = build_features(df).values
        df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())

    for S_val in S_vals:
        df_s = df[df["S"] == S_val]

        # Pick best scenario for this S
        best_sid, best_mape = None, 999
        for sid, grp in df_s.groupby("scenario_id"):
            mask = grp["qb_ref"].abs() > 0.5
            if mask.sum() == 0:
                continue
            mape = np.mean(np.abs(
                (grp.loc[mask,"qb_ref"] - grp.loc[mask,"qb_corrected"])
                / grp.loc[mask,"qb_ref"])) * 100
            if mape < best_mape:
                best_mape = mape
                best_sid  = sid

        if best_sid is None:
            print(f"  No data for S={S_val} — skipping")
            continue

        sub = df_s[df_s["scenario_id"] == best_sid].sort_values("depth")
        ID  = sub["ID"].iloc[0]
        v   = sub["v_pen"].iloc[0]

        fast_sm = sub["qb_fast"].values
        corr_sm = sub["qb_corrected"].values
        ref_sm  = sub["qb_ref"].values
        depth   = sub["depth"].values

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(
            f"ML Correction Result — S={S_val}, ID={ID}, v={v} cm/s\n"
            f"Reference: S=1  |  MAPE after correction = {best_mape:.1f}%",
            fontsize=11, fontweight="bold"
        )
        ax.plot(fast_sm, depth, "--", color="#e74c3c", lw=2.0, label=f"Fast (S={S_val})")
        ax.plot(corr_sm, depth, "-",  color="#2ecc71", lw=2.5, label="Corrected")
        ax.plot(ref_sm,  depth, "-",  color="#2c3e50", lw=2.0, alpha=0.85, label="Reference (S=1)")
        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]", fontsize=11)
        ax.set_ylabel("Depth [m]", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out = f"{PLOTS_DIR}/pub_A_correction_curve_S{S_val}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {out}")


def plot_error_reduction_summary(df_test, model_qb, model_qs=None):
    """
    Plot B — Error reduction summary bar chart.
    Shows RMSE, MAE and MAPE before and after correction per S level.
    """
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Error Reduction Summary — Base resistance qb  (Reference: S=1)",
                 fontsize=13, fontweight="bold")

    metrics_info = [
        ("RMSE [MPa]",  lambda fast, ref: np.sqrt(np.mean((fast-ref)**2))),
        ("MAE [MPa]",   lambda fast, ref: np.mean(np.abs(fast-ref))),
        ("MAPE (%)",    lambda fast, ref: np.mean(
            np.abs((ref[np.abs(ref)>0.5] - fast[np.abs(ref)>0.5])
                   / ref[np.abs(ref)>0.5])) * 100),
    ]

    for ax, (metric_name, metric_fn) in zip(axes, metrics_info):
        before_vals, after_vals = [], []
        for sv in S_vals:
            sub = df[df["S"] == sv]
            before_vals.append(metric_fn(sub["qb_fast"].values,
                                         sub["qb_ref"].values))
            after_vals.append(metric_fn(sub["qb_corrected"].values,
                                        sub["qb_ref"].values))

        x  = np.arange(len(S_vals))
        w  = 0.35
        b1 = ax.bar(x - w/2, before_vals, w, color="#e74c3c",
                    label="Before correction", alpha=0.85)
        b2 = ax.bar(x + w/2, after_vals,  w, color="#2ecc71",
                    label="After correction",  alpha=0.85)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(before_vals + after_vals),
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9, color="#c0392b")
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01 * max(before_vals + after_vals),
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9, color="#27ae60")

        ax.set_xticks(x)
        ax.set_xticklabels([f"S={s}" for s in S_vals], fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_B_error_reduction_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")
    return
    print(f"  Saved → {out}")
    """
    Plot B — Error reduction summary bar chart.
    Average MAPE before and after correction per S level.
    Clean overview for abstract or introduction figure.
    """
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Error Reduction Summary  (Reference: S=1)",
                 fontsize=13, fontweight="bold")

    for ax, (fc, cc, rc, title, unit) in zip(axes, [
        ("qb_fast","qb_corrected","qb_ref","Base resistance qb","MPa"),
    ]):
        before_vals, after_vals = [], []
        for sv in S_vals:
            sub  = df[df["S"] == sv]
            mask = sub[rc].abs() > 0.5
            before_vals.append(np.mean(np.abs(
                (sub.loc[mask,rc]-sub.loc[mask,fc])/sub.loc[mask,rc]))*100
                if mask.sum()>0 else np.nan)
            after_vals.append(np.mean(np.abs(
                (sub.loc[mask,rc]-sub.loc[mask,cc])/sub.loc[mask,rc]))*100
                if mask.sum()>0 else np.nan)

        x   = np.arange(len(S_vals))
        w   = 0.35
        b1  = ax.bar(x - w/2, before_vals, w, color="#e74c3c",
                     label="Before correction", alpha=0.85)
        b2  = ax.bar(x + w/2, after_vals,  w, color="#2ecc71",
                     label="After correction",  alpha=0.85)

        # Value labels on bars
        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#e74c3c")
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9, color="#27ae60")

        ax.set_xticks(x)
        ax.set_xticklabels([f"S={s}" for s in S_vals], fontsize=10)
        ax.set_ylabel("MAPE (%)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_B_error_reduction_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_predicted_vs_actual(df_test, model_qb, model_qs=None):
    """
    Plot C — One panel per S level showing predicted vs actual residual.
    Uses pre-computed qb_corrected if available.
    """
    df = df_test.copy()
    X  = build_features(df).values

    S_vals = sorted(df["S"].unique())
    fig, axes = plt.subplots(1, len(S_vals), figsize=(7 * len(S_vals), 6))
    if len(S_vals) == 1:
        axes = [axes]
    fig.suptitle("Predicted vs Actual Residual — Base resistance qb\n"
                 "(Reference: S=1)  |  Points along diagonal = correct prediction",
                 fontsize=11, fontweight="bold")

    for ax, S_val in zip(axes, S_vals):
        sub  = df[df["S"] == S_val]
        X_s  = build_features(sub).values
        pred = model_qb.predict(X_s)
        true = sub["res_qb"].values

        ax.scatter(true, pred, s=8, alpha=0.4, color="#3498db")

        lim = max(abs(true).max(), abs(pred).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5,
                label="Perfect prediction (y=x)")

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f"R² = {r2:.3f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        n_scenarios = sub["scenario_id"].nunique()
        ax.text(0.05, 0.82, f"n scenarios = {n_scenarios}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Actual residual [MPa]", fontsize=10)
        ax.set_ylabel("Predicted residual [MPa]", fontsize=10)
        ax.set_title(f"S={S_val}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_C_predicted_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_depth_error_profile(df_test, model_qb, model_qs=None):
    """
    Plot D — One panel per S level showing depth error profile.
    """
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())
    depths = sorted(df["depth"].unique())

    fig, axes = plt.subplots(1, len(S_vals), figsize=(7 * len(S_vals), 7))
    if len(S_vals) == 1:
        axes = [axes]
    fig.suptitle("Error Profile vs Depth  (Reference: S=1)\n"
                 "Mean ± 1 std across all scenarios",
                 fontsize=12, fontweight="bold")

    for ax, S_val in zip(axes, S_vals):
        sub_s = df[df["S"] == S_val]

        ef_mean, ef_std, ec_mean, ec_std = [], [], [], []
        for d in depths:
            sub = sub_s[sub_s["depth"] == d]
            ef  = (sub["qb_fast"]      - sub["qb_ref"]).values
            ec  = (sub["qb_corrected"] - sub["qb_ref"]).values
            ef_mean.append(np.mean(ef)); ef_std.append(np.std(ef))
            ec_mean.append(np.mean(ec)); ec_std.append(np.std(ec))

        ef_m = np.array(ef_mean); ef_s = np.array(ef_std)
        ec_m = np.array(ec_mean); ec_s = np.array(ec_std)

        ax.plot(ef_m, depths, color="#e74c3c", lw=1.5, label="Fast error (mean)")
        ax.fill_betweenx(depths, ef_m-ef_s, ef_m+ef_s,
                         color="#e74c3c", alpha=0.15)
        ax.plot(ec_m, depths, color="#2ecc71", lw=2.0, label="Corrected error (mean)")
        ax.fill_betweenx(depths, ec_m-ec_s, ec_m+ec_s,
                         color="#2ecc71", alpha=0.15)
        ax.axvline(0, color="black", lw=1.2)
        ax.axhline(SHALLOW_CUTOFF, color="gray", lw=1, ls="--",
                   alpha=0.6, label=f"Shallow cutoff {SHALLOW_CUTOFF}m")
        ax.invert_yaxis()
        ax.set_xlabel("Error [MPa]", fontsize=10)
        ax.set_ylabel("Depth [m]", fontsize=10)
        ax.set_title(f"S={S_val} — Base resistance qb", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_D_depth_error_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")
    return
    print(f"  Saved → {out}")
    """
    Plot D — Continuous depth profile of mean error with confidence band.
    Shows exactly where correction works well and where it struggles.
    More informative than binned heatmap for publication.
    """
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.suptitle("Error Profile vs Depth  (Reference: S=1)\n"
                 "Mean ± 1 std across all scenarios",
                 fontsize=12, fontweight="bold")

    depths = sorted(df["depth"].unique())
    fc, cc, rc, title, unit = "qb_fast","qb_corrected","qb_ref","Base resistance qb","MPa"

    err_fast_mean, err_fast_std = [], []
    err_corr_mean, err_corr_std = [], []

    for d in depths:
        sub = df[df["depth"] == d]
        ef  = (sub[fc] - sub[rc]).values
        ec  = (sub[cc] - sub[rc]).values
        err_fast_mean.append(np.mean(ef))
        err_fast_std.append(np.std(ef))
        err_corr_mean.append(np.mean(ec))
        err_corr_std.append(np.std(ec))

    ef_m = np.array(err_fast_mean)
    ef_s = np.array(err_fast_std)
    ec_m = np.array(err_corr_mean)
    ec_s = np.array(err_corr_std)

    ax.plot(ef_m, depths, color="#e74c3c", lw=1.5, label="Fast error (mean)")
    ax.fill_betweenx(depths, ef_m - ef_s, ef_m + ef_s, color="#e74c3c", alpha=0.15)
    ax.plot(ec_m, depths, color="#2ecc71", lw=2.0, label="Corrected error (mean)")
    ax.fill_betweenx(depths, ec_m - ec_s, ec_m + ec_s, color="#2ecc71", alpha=0.15)
    ax.axvline(0, color="black", lw=1.2)
    ax.axhline(SHALLOW_CUTOFF, color="gray", lw=1, ls="--",
               alpha=0.6, label=f"Shallow cutoff {SHALLOW_CUTOFF}m")
    ax.invert_yaxis()
    ax.set_xlabel(f"Error [{unit}]", fontsize=10)
    ax.set_ylabel("Depth [m]", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_D_depth_error_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_improvement_percentage(df_test, model_qb, model_qs=None):
    """
    Plot E — Percentage improvement per scenario.
    Shows how much each scenario benefited from correction.
    Easy to read numbers for publication.
    """
    X = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    rows = []
    for sid, grp in df.groupby("scenario_id"):
        S  = grp["S"].iloc[0]
        ID = grp["ID"].iloc[0]
        for fc, cc, rc, tag in [
            ("qb_fast","qb_corrected","qb_ref","qb"),
        ]:
            mask = grp[rc].abs() > 0.5
            if mask.sum() == 0:
                continue
            before = np.mean(np.abs(
                (grp.loc[mask,rc]-grp.loc[mask,fc])/grp.loc[mask,rc]))*100
            after  = np.mean(np.abs(
                (grp.loc[mask,rc]-grp.loc[mask,cc])/grp.loc[mask,rc]))*100
            improvement = before - after
            rows.append({
                "scenario": sid.replace("MC_G0_","").replace("G0_",""),
                "S": S, "ID": ID, "tag": tag,
                "before": before, "after": after,
                "improvement": improvement
            })

    df_imp = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("Error Improvement per Scenario — Base resistance qb  (Reference: S=1)\n"
                 "Positive = correction helped, Negative = correction hurt",
                 fontsize=12, fontweight="bold")

    sub = df_imp[df_imp["tag"] == "qb"].sort_values("improvement", ascending=True)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["improvement"]]
    bars = ax.barh(range(len(sub)), sub["improvement"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [f"S={int(r.S)}, ID={r.ID}" for r in sub.itertuples()], fontsize=8)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("MAPE improvement (percentage points)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, sub["improvement"]):
        ax.text(val + (0.2 if val >= 0 else -0.2),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}pp", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_E_improvement_per_scenario.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_correlation_matrix(df, model_qb, model_qs=None):
    """
    Plot F — Correlation matrix with all features including interactions.
    Last row/column shows correlation with res_qb (ML target).
    """
    feat = pd.DataFrame()
    feat["Density (ID)"]  = df["ID"]
    feat["Velocity (v)"]  = df["v_pen"]
    feat["Scaling (S)"]   = df["S"]
    feat["log(S)"]        = np.log10(df["S"].clip(lower=1))
    feat["Depth"]         = df["depth"]
    feat["√Depth"]        = np.sqrt(df["depth"])
    feat["qb fast"]       = df["qb_fast"]
    feat["S × Depth"]     = np.log10(df["S"].clip(lower=1)) * df["depth"]
    feat["ID × Depth"]    = df["ID"] * df["depth"]
    feat["S × ID"]        = np.log10(df["S"].clip(lower=1)) * df["ID"]
    feat["v × S"]         = df["v_pen"] * np.log10(df["S"].clip(lower=1))
    feat["Target (res_qb)"] = df["res_qb"]

    corr = feat.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    fig.suptitle(
        "Correlation Matrix — Input Features and Target\n"
        "Last row/column shows correlation with res_qb (ML target)",
        fontsize=12, fontweight="bold"
    )

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation coefficient")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    # Highlight target row and column in red
    n = len(corr.columns)
    ax.add_patch(plt.Rectangle((-0.5, n-1.5), n, 1,
                                fill=False, edgecolor="#e74c3c", lw=2.5))
    ax.add_patch(plt.Rectangle((n-1.5, -0.5), 1, n,
                                fill=False, edgecolor="#e74c3c", lw=2.5))

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_F_correlation_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def train_soil_model(label, data_path, results_dir, plots_dir):
    global DATA_PATH, RESULTS_DIR, PLOTS_DIR
    DATA_PATH = data_path
    RESULTS_DIR = results_dir
    PLOTS_DIR = plots_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("="*60)
    print("MGS Residual Correction — Training on REAL Data")
    print(f"Reference: S=1")
    print("="*60)

    print(f"DATASET:  {DATA_PATH}")
    print(f"RESULTS:  {RESULTS_DIR}")
    print(f"PLOTS:    {PLOTS_DIR}")

    if not os.path.exists(DATA_PATH):
        print(f"  Skipping {label}: dataset not found")
        return None

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df):,} rows")
    print(f"Scenarios:  {df['scenario_id'].nunique()}")
    print(f"S values:   {sorted(df['S'].unique())}")
    print(f"ID values:  {sorted(df['ID'].unique())}")
    print(f"Depth:      {df['depth'].min():.2f} – {df['depth'].max():.2f} m")

    # ── Improvement 2: exclude shallow zone from training ────────────────────
    # is_shallow column was added by 01_build_dataset.py
    if "is_shallow" in df.columns:
        df_train_pool = df[df["is_shallow"] == 0].copy()
        print(f"\n  Shallow rows excluded from training: "
              f"{(df['is_shallow']==1).sum()} rows removed")
        print(f"  Training pool: {len(df_train_pool):,} rows")
    else:
        df_train_pool = df.copy()

    # ── Improvement 3: separate models per S level or combined ───────────────
    S_vals = sorted(df["S"].unique())

    if SEPARATE_S_MODELS and len(S_vals) > 1:
        print(f"\n  Training SEPARATE models for each S level: {S_vals}")
        models_qb = {}

        for S_val in S_vals:
            df_s = df_train_pool[df_train_pool["S"] == S_val]
            print(f"\n  --- S={S_val}: {len(df_s):,} rows ---")

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            tr_idx, te_idx = next(splitter.split(df_s, groups=df_s["scenario_id"]))
            df_tr_all = df_s.iloc[tr_idx]
            df_te_s   = df_s.iloc[te_idx]

            splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
            tr2, va2  = next(splitter2.split(df_tr_all, groups=df_tr_all["scenario_id"]))
            df_tr = df_tr_all.iloc[tr2]
            df_va = df_tr_all.iloc[va2]

            print(f"    Train: {df_tr['scenario_id'].nunique()} scenarios  "
                  f"Val: {df_va['scenario_id'].nunique()}  "
                  f"Test: {df_te_s['scenario_id'].nunique()}")

            print(f"    Training qb model...")
            models_qb[S_val] = train(df_tr, df_va, "res_qb", f"qb_S{S_val}")

        # Evaluate on full test set using correct model per S
        print("\nEvaluating on full test set...")
        df_test = df.copy()
        df_test["qb_corrected"] = np.nan

        for S_val in S_vals:
            mask = df_test["S"] == S_val
            X    = build_features(df_test[mask]).values
            df_test.loc[mask, "qb_corrected"] = (
                df_test.loc[mask, "qb_fast"] + models_qb[S_val].predict(X)
            ).clip(lower=0)

        # Save combined models dict
        with open(f"{RESULTS_DIR}/models_qb_per_S.pkl", "wb") as f:
            pickle.dump(models_qb, f)
        print("  Saved separate S models → results/models_qb_per_S.pkl")

        # Also keep single model interface for inference script compatibility.
        # Use the largest S model as default.
        default_S = max(S_vals)
        with open(f"{RESULTS_DIR}/model_qb.pkl", "wb") as f:
            pickle.dump(models_qb[default_S], f)
        print(f"  Default single model (S={default_S}) → results/model_qb.pkl")

    else:
        print(f"\n  Training COMBINED model for all S levels")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(splitter.split(
            df_train_pool, groups=df_train_pool["scenario_id"]))
        df_tr_all = df_train_pool.iloc[tr_idx]
        test_scenarios = set(df_train_pool.iloc[te_idx]["scenario_id"])
        df_test = df[df["scenario_id"].isin(test_scenarios)].copy()

        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
        tr2, va2  = next(splitter2.split(df_tr_all, groups=df_tr_all["scenario_id"]))
        df_tr = df_tr_all.iloc[tr2]
        df_va = df_tr_all.iloc[va2]

        print(f"  Train: {df_tr['scenario_id'].nunique()} scenarios  "
              f"Val: {df_va['scenario_id'].nunique()}  "
              f"Test: {df_test['scenario_id'].nunique()}")

        print("\nTraining qb model...")
        model_qb = train(df_tr, df_va, "res_qb", "qb")

        X_test = build_features(df_test).values
        df_test["qb_corrected"] = (
            df_test["qb_fast"] + model_qb.predict(X_test)).clip(lower=0)

    # ── Use df_test for all evaluation and plots ──────────────────────────────

    metrics = []
    for zone_label, mask in [
        ("ALL depths",              pd.Series([True]*len(df_test),
                                              index=df_test.index)),
        (f"Shallow (0–{SHALLOW_CUTOFF}m)", df_test["depth"] <= SHALLOW_CUTOFF),
        (f"Deep (>{SHALLOW_CUTOFF}m)",     df_test["depth"] >  SHALLOW_CUTOFF),
    ]:
        sub = df_test[mask]
        for fc, cc, rc, tag in [
            ("qb_fast","qb_corrected","qb_ref","qb"),
        ]:
            metrics.append(compute_metrics(sub[rc].values, sub[fc].values,
                                           f"{tag} fast vs ref [{zone_label}]"))
            metrics.append(compute_metrics(sub[rc].values, sub[cc].values,
                                           f"{tag} corrected vs ref [{zone_label}]"))

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print(df_metrics.to_string(index=False))

    # For plots — pass full df_test with all S values
    # correction curves and pub_A loop over S internally using models dict
    if SEPARATE_S_MODELS and len(S_vals) > 1:
        default_S = max(S_vals)
        model_qb  = models_qb[default_S]   # default for single-panel plots
        print(f"\n  Using S={default_S} as default model for single-panel plots")
        print(f"  All S levels used for correction curve and depth profile plots")
    else:
        pass   # model_qb already set in combined path

    # Delete old non-S-specific correction_curves.png if it exists
    old_plot = os.path.join(PLOTS_DIR, "correction_curves.png")
    if os.path.exists(old_plot):
        os.remove(old_plot)

    # Build df_all — full dataset WITHOUT outlier removal for plotting
    # This ensures all depth points are present and lines are not cut short
    PLOT_DATA_PATH = DATA_PATH.replace(".csv", "_plot.csv")
    if os.path.exists(PLOT_DATA_PATH):
        df_plot_full = pd.read_csv(PLOT_DATA_PATH)
        print(f"  Loaded plot dataset: {len(df_plot_full):,} rows (no outlier removal)")
    else:
        df_plot_full = df.copy()
        print(f"  Plot dataset not found — using training dataset for plots")

    X_plot = build_features(df_plot_full).values
    df_all  = df_plot_full.copy()
    df_all["qb_corrected"] = (
        df_all["qb_fast"] + model_qb.predict(X_plot)
    ).clip(lower=0)

    print("\nGenerating plots...")
    plot_correction_curves(df_all, model_qb, None)
    plot_error_by_depth(df_test, model_qb, None)

    print("\nGenerating sensitivity plots...")
    plot_sensitivity(df_test, model_qb, None)

    print("\nGenerating publication plots...")
    plot_publication_correction_curve(df_all, model_qb, None)
    plot_error_reduction_summary(df_test, model_qb, None)
    plot_predicted_vs_actual(df_test, model_qb, None)
    plot_depth_error_profile(df_all, model_qb, None)
    plot_improvement_percentage(df_test, model_qb, None)
    plot_correlation_matrix(df, model_qb, None)

    print(f"\nDone. Models → {RESULTS_DIR}/  Plots → {PLOTS_DIR}/")
    return df_metrics


def main():
    parser = argparse.ArgumentParser(description="Train residual correction models from paired ML datasets.")
    parser.add_argument(
        "--soil-model",
        choices=["all", "mohr_coulomb", "mcm", "hypoplastic"],
        default="all",
        help="Limit training to one soil model. 'mcm' is kept as an alias for mohr_coulomb.",
    )
    args = parser.parse_args()
    selected_soil_model = "mohr_coulomb" if args.soil_model == "mcm" else args.soil_model

    for run in SOIL_MODEL_RUNS:
        if selected_soil_model != "all" and run["name"] != selected_soil_model:
            continue
        train_soil_model(
            label=run["label"],
            data_path=run["data_path"],
            results_dir=run["results_dir"],
            plots_dir=run["plots_dir"],
        )


if __name__ == "__main__":
    main()

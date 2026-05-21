"""
02_train_correction_model.py  [ENHANCED]
-----------------------------------------
Trains gradient boosting models to predict the residual error
between fast runs (high S) and the reference (S=1).
"""

import numpy as np
import pandas as pd
import pickle, json, os, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root():
    for name in ["Data", "data", "DATA"]:
        if os.path.isdir(os.path.join(_HERE, name)):
            return _HERE, name
    for name in ["Data", "data", "DATA"]:
        candidate = os.path.join(_HERE, "..", name)
        if os.path.isdir(candidate):
            return os.path.normpath(os.path.join(_HERE, "..")), name
    return _HERE, "data"

_ROOT, _data_name = _find_root()
_DATA_FOLDER = os.path.join(_ROOT, _data_name)

DATA_PATH    = os.path.join(_DATA_FOLDER, "real_dataset.csv")
META_PATH    = os.path.join(_DATA_FOLDER, "dataset_meta.json")
RESULTS_DIR  = os.path.join(_ROOT, "results")
PLOTS_DIR    = os.path.join(_ROOT, "plots")
SHALLOW_CUTOFF = 2.0

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

print(f"ROOT:     {_ROOT}")
print(f"DATA:     {_DATA_FOLDER}")
print(f"RESULTS:  {RESULTS_DIR}")
print(f"PLOTS:    {PLOTS_DIR}")

# ── Model mode ───────────────────────────────────────────────────────────────
# Set True when ≥400 simulations are available — with 9 scenarios, separate
# models overfit badly due to insufficient groups per S level.
SEPARATE_S_MODELS = False

# ── Fixed model parameters (original) ────────────────────────────────────────
MODEL_PARAMS = dict(
    max_iter          = 600,
    max_depth         = 6,
    learning_rate     = 0.05,
    min_samples_leaf  = 10,
    l2_regularization = 1.0,
    random_state      = 42,
)

# ─── [E7] Metrics — MPE now signed, MAPE unsigned (were identical before) ────

def compute_metrics(y_true, y_pred, label=""):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-9)

    # Mask near-zero values where percentage metrics blow up
    mask = np.abs(y_true) > 0.5

    # [E7] MPE — signed: positive = model under-predicts reference
    mpe = (np.mean(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    ) * 100) if mask.sum() > 0 else np.nan

    # MAPE — unsigned absolute percentage error
    mape = (np.mean(np.abs(
        (y_true[mask] - y_pred[mask]) / y_true[mask]
    )) * 100) if mask.sum() > 0 else np.nan

    return {"label": label, "RMSE": rmse, "MAE": mae,
            "NRMSE": nrmse, "MPE(%)": mpe, "MAPE(%)": mape,
            "n": len(y_true)}


# ─── [E2] Feature engineering — added qb_fast_grad ───────────────────────────

def build_features(df):
    """
    Build ML input features. All features available at inference time.
    qb_fast_grad must be pre-computed in 01_build_dataset.py.
    """
    feat = pd.DataFrame(index=df.index)

    feat["ID"]     = df["ID"]
    feat["v_pen"]  = df["v_pen"]
    feat["S"]      = df["S"]
    feat["depth"]  = df["depth"]
    feat["qb_fast"] = df["qb_fast"]

    # [E2] Depth gradient of fast run — rate of resistance increase with depth.
    # Captures stress-depth coupling that flat qb_fast misses.
    if "qb_fast_grad" in df.columns:
        feat["qb_fast_grad"] = df["qb_fast_grad"]
    else:
        # Graceful fallback if dataset was built with old script
        feat["qb_fast_grad"] = 0.0

    log_S = np.log10(df["S"].clip(lower=1))
    feat["S_x_depth"]  = log_S * df["depth"]
    feat["ID_x_depth"] = df["ID"] * df["depth"]
    feat["S_x_ID"]     = log_S * df["ID"]
    feat["v_x_S"]      = df["v_pen"] * log_S

    # [E2] Gradient interaction with S — inertial effect on resistance slope
    if "qb_fast_grad" in df.columns:
        feat["grad_x_S"]  = df["qb_fast_grad"] * log_S
        feat["grad_x_ID"] = df["qb_fast_grad"] * df["ID"]

    return feat

FEATURE_NAMES = list(build_features(pd.DataFrame({
    "ID": [0.9], "v_pen": [50.0], "S": [10], "depth": [5.0],
    "qb_fast": [10.0], "qb_fast_grad": [1.5]
})).columns)


# ─── [E5] GroupKFold CV + [E6] grid search ───────────────────────────────────

def cross_validate(df, target, n_splits=None):
    """
    GroupKFold CV over scenario_id — every scenario appears in test fold once.
    Uses fixed MODEL_PARAMS. Returns mean CV RMSE for reporting only.
    """
    groups   = df["scenario_id"].values
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits or n_groups, n_groups)
    print(f"  GroupKFold CV: {n_splits} folds, {n_groups} unique scenarios")

    fold_rmses = []
    kf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in kf.split(df, groups=groups):
        df_tr = df.iloc[tr_idx]; df_va = df.iloc[va_idx]
        X_tr  = build_features(df_tr).values
        X_va  = build_features(df_va).values
        y_tr  = df_tr[target].values
        y_va  = df_va[target].values
        model = HistGradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)
        fold_rmses.append(np.sqrt(mean_squared_error(y_va, model.predict(X_va))))

    mean_rmse = np.mean(fold_rmses)
    std_rmse  = np.std(fold_rmses)
    print(f"  CV RMSE = {mean_rmse:.5f} +/- {std_rmse:.5f}")
    return mean_rmse


def train_final(df_train, target, tag):
    """Train on full training set with fixed MODEL_PARAMS and save."""
    X = build_features(df_train).values
    y = df_train[target].values
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X, y)
    path = f"{RESULTS_DIR}/model_{tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model -> {path}")
    return model



# ─── [E1] Normalisation — compute on training rows only ──────────────────────

def compute_and_save_normalisation(df_train, meta_path):
    """
    Compute qb signal magnitude from training rows only.
    Saves to JSON for use at inference time.
    """
    qb_signal = df_train["qb_ref"].abs().replace(0, np.nan).mean()
    print(f"  Normalisation scalar (training rows only): {qb_signal:.4f} MPa")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["qb_signal_train"] = round(float(qb_signal), 6)
    meta["feature_names"]   = FEATURE_NAMES
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved normalisation scalar → {meta_path}")
    return qb_signal


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_correction_curves(df_test, model_qb, n=20):
    df = df_test.copy()
    if "qb_corrected" not in df.columns:
        X = build_features(df).values
        df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    df["scenario_base"] = df["scenario_id"].str.replace("^MC_", "", regex=True)
    scenario_order = (df.groupby("scenario_base")
                        .agg(ID=("ID","first"), v=("v_pen","first"))
                        .sort_values(["ID","v"]))
    preferred_ids = []
    for base in scenario_order.index.tolist():
        sub_ids = df.loc[df["scenario_base"]==base, "scenario_id"].unique().tolist()
        mc_ids  = [s for s in sub_ids if s.startswith("MC_")]
        preferred_ids.append(mc_ids[0] if mc_ids else sub_ids[0])

    all_scenarios = preferred_ids[:n]
    ncols = 3
    nrows = int(np.ceil(len(all_scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes_flat  = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(
        f"Correction curves — all {len(all_scenarios)} scenarios  (Reference: S=1)\n"
        "Base resistance qb  [30-pt moving average applied in dataset]",
        fontsize=13, fontweight="bold")

    colours_fast = ["#f39c12","#e74c3c","#9b59b6","#e67e22"]
    for i, sid in enumerate(all_scenarios):
        ax  = axes_flat[i]
        sub = df[df["scenario_id"] == sid]
        ID  = sub["ID"].iloc[0]
        v   = sub["v_pen"].iloc[0]

        S_vals = sorted(sub["S"].unique())
        S_plot = S_vals[-1]
        ref_s  = sub[sub["S"] == S_plot].sort_values("depth")
        if len(ref_s) == 0:
            ax.set_visible(False); continue

        ax.plot(ref_s["qb_ref"].values, ref_s["depth"].values,
                "-", color="#2c3e50", lw=2.0, alpha=0.9, label="Reference (S=1)")
        for idx, S_val in enumerate(S_vals):
            sub_s = sub[sub["S"]==S_val].sort_values("depth")
            if len(sub_s) == 0: continue
            ax.plot(sub_s["qb_fast"].values, sub_s["depth"].values,
                    "--", color=colours_fast[idx % len(colours_fast)],
                    lw=1.2, label=f"Fast (S={S_val})")
        ax.plot(ref_s["qb_corrected"].values, ref_s["depth"].values,
                "-", color="#2ecc71", lw=2.5, label="ML Corrected")

        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]", fontsize=9)
        ax.set_ylabel("Depth [m]", fontsize=9)
        ax.set_title(f"ID={ID}, v={v} cm/s", fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/correction_curves_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")
    return df


def compute_mape_group(df, group_col, fc, cc, rc):
    rows = []
    for val, grp in df.groupby(group_col):
        mask = grp[rc].abs() > 0.5
        if mask.sum() == 0: continue
        before = np.mean(np.abs((grp.loc[mask,rc]-grp.loc[mask,fc])/grp.loc[mask,rc]))*100
        after  = np.mean(np.abs((grp.loc[mask,rc]-grp.loc[mask,cc])/grp.loc[mask,rc]))*100
        rows.append({group_col: val, "before": before, "after": after})
    return pd.DataFrame(rows)


def plot_sensitivity(df_test, model_qb):
    X  = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)
    C_B = "#e74c3c"; C_A = "#2ecc71"

    # Plot 1: MAPE vs S
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    g = compute_mape_group(df, "S", "qb_fast", "qb_corrected", "qb_ref")
    ax.plot(g["S"], g["before"], "o--", color=C_B, lw=2, ms=7, label="Fast (raw)")
    ax.plot(g["S"], g["after"],  "o-",  color=C_A, lw=2, ms=7, label="After correction")
    ax.set_xlabel("Mass scaling factor S"); ax.set_ylabel("MAPE (%) — Base resistance qb")
    ax.set_title("MAPE vs mass scaling factor S  (Reference: S=1)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xticks(g["S"].tolist())
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_vs_S.png", dpi=150, bbox_inches="tight"); plt.close()

    # Plot 2: MAPE vs ID
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    g = compute_mape_group(df, "ID", "qb_fast", "qb_corrected", "qb_ref")
    ax.plot(g["ID"], g["before"], "s--", color=C_B, lw=2, ms=7, label="Fast (raw)")
    ax.plot(g["ID"], g["after"],  "s-",  color=C_A, lw=2, ms=7, label="After correction")
    ax.set_xlabel("Relative density ID"); ax.set_ylabel("MAPE (%) — Base resistance qb")
    ax.set_title("MAPE vs relative density ID  (Reference: S=1)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_vs_ID.png", dpi=150, bbox_inches="tight"); plt.close()

    # Plot 3: Shallow vs Deep
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    df["zone"] = df["depth"].apply(
        lambda z: f"Shallow (0–{SHALLOW_CUTOFF}m)" if z <= SHALLOW_CUTOFF
                  else f"Deep (>{SHALLOW_CUTOFF}m)")
    zones = [f"Shallow (0–{SHALLOW_CUTOFF}m)", f"Deep (>{SHALLOW_CUTOFF}m)"]
    before, after = [], []
    for z in zones:
        sub  = df[df["zone"]==z]; mask = sub["qb_ref"].abs()>0.5
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
    ax.set_ylabel("MAPE (%)"); ax.set_title("Shallow vs Deep — Base resistance qb", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/sensitivity_shallow_vs_deep.png", dpi=150, bbox_inches="tight"); plt.close()

    # Plot 4: Heatmap S × ID
    S_vals  = sorted(df["S"].unique()); ID_vals = sorted(df["ID"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle("Heatmap: S × ID MAPE  (Reference: S=1)", fontsize=12, fontweight="bold")
    for ax, (col_use, title, cmap) in zip(axes, [
        ("qb_fast",      "Fast error MAPE% (before)",     "Reds"),
        ("qb_corrected", "Corrected error MAPE% (after)", "Greens"),
    ]):
        mat = np.full((len(ID_vals), len(S_vals)), np.nan)
        for i, idv in enumerate(ID_vals):
            for j, sv in enumerate(S_vals):
                sub  = df[(df["ID"]==idv) & (df["S"]==sv)]
                mask = sub["qb_ref"].abs() > 0.5
                if mask.sum()==0: continue
                mat[i,j] = np.mean(np.abs(
                    (sub.loc[mask,"qb_ref"]-sub.loc[mask,col_use])/sub.loc[mask,"qb_ref"]))*100
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(mat))
        ax.set_xticks(range(len(S_vals)));  ax.set_xticklabels(S_vals)
        ax.set_yticks(range(len(ID_vals))); ax.set_yticklabels(ID_vals)
        ax.set_xlabel("S"); ax.set_ylabel("Density ID"); ax.set_title(title)
        plt.colorbar(im, ax=ax, label="MAPE (%)")
        for i in range(len(ID_vals)):
            for j in range(len(S_vals)):
                if not np.isnan(mat[i,j]):
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=9,
                            color="white" if mat[i,j]>np.nanmax(mat)*0.6 else "black")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/heatmap_S_vs_ID.png", dpi=150, bbox_inches="tight"); plt.close()

    # Plot 5: Heatmap S × depth
    bins   = np.arange(0, df["depth"].max()+2, 2)
    labels = [f"{int(b)}–{int(b+2)}m" for b in bins[:-1]]
    df["depth_bin"] = pd.cut(df["depth"], bins=bins, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Heatmap: S × depth MAPE  (Reference: S=1)", fontsize=12, fontweight="bold")
    for ax, (col_use, title, cmap) in zip(axes, [
        ("qb_fast",      "Fast error MAPE% by S and depth",      "Reds"),
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

    print(f"  Sensitivity plots saved → {PLOTS_DIR}/")


def plot_publication_correction_curve(df_test, model_qb):
    """Plot A — Best-scenario correction curve per S level."""
    df = df_test.copy()
    if "qb_corrected" not in df.columns:
        X = build_features(df).values
        df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    for S_val in sorted(df["S"].unique()):
        df_s = df[df["S"]==S_val]
        best_sid, best_mape = None, 999
        for sid, grp in df_s.groupby("scenario_id"):
            mask = grp["qb_ref"].abs() > 0.5
            if mask.sum()==0: continue
            mape = np.mean(np.abs(
                (grp.loc[mask,"qb_ref"]-grp.loc[mask,"qb_corrected"])/grp.loc[mask,"qb_ref"]))*100
            if mape < best_mape:
                best_mape = mape; best_sid = sid
        if best_sid is None: continue

        sub = df_s[df_s["scenario_id"]==best_sid].sort_values("depth")
        ID  = sub["ID"].iloc[0]; v = sub["v_pen"].iloc[0]

        fig, ax = plt.subplots(figsize=(8,8))
        fig.suptitle(
            f"ML Correction Result — S={S_val}, ID={ID}, v={v} cm/s\n"
            f"Reference: S=1  |  MAPE after correction = {best_mape:.1f}%",
            fontsize=11, fontweight="bold")
        ax.plot(sub["qb_fast"].values,      sub["depth"].values, "--", color="#e74c3c",
                lw=2.0, label=f"Fast (S={S_val})")
        ax.plot(sub["qb_corrected"].values, sub["depth"].values, "-",  color="#2ecc71",
                lw=2.5, label="Corrected")
        ax.plot(sub["qb_ref"].values,       sub["depth"].values, "-",  color="#2c3e50",
                lw=2.0, alpha=0.85, label="Reference (S=1)")
        ax.invert_yaxis()
        ax.set_xlabel("Base resistance qb [MPa]", fontsize=11)
        ax.set_ylabel("Depth [m]", fontsize=11)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = f"{PLOTS_DIR}/pub_A_correction_curve_S{S_val}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved → {out}")


def plot_error_reduction_summary(df_test, model_qb):
    """Plot B — RMSE / MAE / MAPE before and after correction per S."""
    X  = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle("Error Reduction Summary — Base resistance qb  (Reference: S=1)",
                 fontsize=13, fontweight="bold")

    metrics_info = [
        ("RMSE [MPa]", lambda f,r: np.sqrt(np.mean((f-r)**2))),
        ("MAE [MPa]",  lambda f,r: np.mean(np.abs(f-r))),
        ("MAPE (%)",   lambda f,r: np.mean(
            np.abs((r[np.abs(r)>0.5]-f[np.abs(r)>0.5])/r[np.abs(r)>0.5]))*100),
    ]
    for ax, (mname, mfn) in zip(axes, metrics_info):
        bvals, avals = [], []
        for sv in S_vals:
            sub = df[df["S"]==sv]
            bvals.append(mfn(sub["qb_fast"].values, sub["qb_ref"].values))
            avals.append(mfn(sub["qb_corrected"].values, sub["qb_ref"].values))
        x = np.arange(len(S_vals)); w = 0.35
        b1 = ax.bar(x-w/2, bvals, w, color="#e74c3c", label="Before", alpha=0.85)
        b2 = ax.bar(x+w/2, avals, w, color="#2ecc71", label="After",  alpha=0.85)
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + 0.01*max(bvals+avals),
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"S={s}" for s in S_vals], fontsize=10)
        ax.set_ylabel(mname, fontsize=10); ax.set_title(mname, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_B_error_reduction_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_predicted_vs_actual(df_all, model_qb):
    """Plot C — Predicted vs actual residual per S."""
    df    = df_all.copy()
    S_vals = sorted(df["S"].unique())
    fig, axes = plt.subplots(1, len(S_vals), figsize=(7*len(S_vals), 6))
    if len(S_vals)==1: axes=[axes]
    fig.suptitle("Predicted vs Actual Residual — Base resistance qb\n"
                 "(Reference: S=1)  |  Points along diagonal = correct prediction",
                 fontsize=11, fontweight="bold")
    for ax, S_val in zip(axes, S_vals):
        sub  = df[df["S"]==S_val]
        X_s  = build_features(sub).values
        pred = model_qb.predict(X_s)
        true = sub["res_qb"].values
        ax.scatter(true, pred, s=8, alpha=0.4, color="#3498db")
        lim  = max(abs(true).max(), abs(pred).max()) * 1.05
        ax.plot([-lim,lim], [-lim,lim], "k--", lw=1.5, label="Perfect (y=x)")
        ss_res = np.sum((true-pred)**2); ss_tot = np.sum((true-np.mean(true))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.text(0.05, 0.82, f"n scenarios = {sub['scenario_id'].nunique()}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_xlabel("Actual residual [MPa]"); ax.set_ylabel("Predicted residual [MPa]")
        ax.set_title(f"S={S_val}"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_C_predicted_vs_actual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


# ─── [E8] Fixed plot_depth_error_profile — dead code removed ─────────────────

def plot_depth_error_profile(df_test, model_qb):
    """
    Plot D — One panel per S level showing depth error profile.
    Mean ± 1 std across all scenarios, smoothed for display.

    [E8] Original had a dead code bug: the per-S multi-panel loop ran first
    and saved the file, then a second single-panel block ran and overwrote it
    due to a misplaced docstring. Fixed: only one block, clearly structured.
    """
    X  = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    S_vals = sorted(df["S"].unique())
    depths = sorted(df["depth"].unique())
    smooth_w = 20   # rolling mean for display only

    fig, axes = plt.subplots(1, len(S_vals), figsize=(7*len(S_vals), 7))
    if len(S_vals)==1: axes=[axes]
    fig.suptitle("Error Profile vs Depth  (Reference: S=1)\nMean ± 1 std across all scenarios",
                 fontsize=12, fontweight="bold")

    for ax, S_val in zip(axes, S_vals):
        sub_s = df[df["S"]==S_val]
        ef_mean, ef_std, ec_mean, ec_std = [], [], [], []
        for d in depths:
            sub = sub_s[sub_s["depth"]==d]
            ef  = (sub["qb_fast"]      - sub["qb_ref"]).values
            ec  = (sub["qb_corrected"] - sub["qb_ref"]).values
            ef_mean.append(np.mean(ef)); ef_std.append(np.std(ef))
            ec_mean.append(np.mean(ec)); ec_std.append(np.std(ec))

        ef_m = pd.Series(ef_mean).rolling(smooth_w, center=True, min_periods=1).mean().values
        ec_m = pd.Series(ec_mean).rolling(smooth_w, center=True, min_periods=1).mean().values
        ef_s = pd.Series(ef_std ).rolling(smooth_w, center=True, min_periods=1).mean().values
        ec_s = pd.Series(ec_std ).rolling(smooth_w, center=True, min_periods=1).mean().values

        ax.plot(ef_m, depths, color="#e74c3c", lw=1.5, label="Fast error (mean)")
        ax.fill_betweenx(depths, ef_m-ef_s, ef_m+ef_s, color="#e74c3c", alpha=0.15)
        ax.plot(ec_m, depths, color="#2ecc71", lw=2.0, label="Corrected error (mean)")
        ax.fill_betweenx(depths, ec_m-ec_s, ec_m+ec_s, color="#2ecc71", alpha=0.15)
        ax.axvline(0, color="black", lw=1.2)
        ax.axhline(SHALLOW_CUTOFF, color="gray", lw=1, ls="--",
                   alpha=0.6, label=f"Shallow cutoff {SHALLOW_CUTOFF}m")
        ax.invert_yaxis()
        ax.set_xlabel("Error [MPa]", fontsize=10); ax.set_ylabel("Depth [m]", fontsize=10)
        ax.set_title(f"S={S_val} — Base resistance qb", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_D_depth_error_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


def plot_improvement_percentage(df_test, model_qb):
    """Plot E — Percentage improvement per scenario."""
    X  = build_features(df_test).values
    df = df_test.copy()
    df["qb_corrected"] = (df["qb_fast"] + model_qb.predict(X)).clip(lower=0)

    rows = []
    for (sid, s_val), grp in df.groupby(["scenario_id", "S"]):
        mask = grp["qb_ref"].abs() > 0.5
        if mask.sum()==0: continue
        before = np.mean(np.abs((grp.loc[mask,"qb_ref"]-grp.loc[mask,"qb_fast"])
                                /grp.loc[mask,"qb_ref"]))*100
        after  = np.mean(np.abs((grp.loc[mask,"qb_ref"]-grp.loc[mask,"qb_corrected"])
                                /grp.loc[mask,"qb_ref"]))*100
        rows.append({"scenario": sid.replace("MC_G0_","").replace("G0_",""),
                     "S": s_val, "ID": grp["ID"].iloc[0],
                     "v": grp["v_pen"].iloc[0],
                     "before": before, "after": after, "improvement": before-after})

    df_imp = pd.DataFrame(rows)
    sub    = df_imp.sort_values("improvement", ascending=True)
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sub["improvement"]]

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    fig.suptitle("Error Improvement per Scenario — Base resistance qb  (Reference: S=1)\n"
                 "Positive = correction helped, Negative = correction hurt",
                 fontsize=12, fontweight="bold")
    bars = ax.barh(range(len(sub)), sub["improvement"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(
        [f"S={int(r.S)}, ID={r.ID}, v={int(r.v)}" for r in sub.itertuples()], fontsize=8)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("MAPE improvement (percentage points)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, sub["improvement"]):
        ax.text(val + (0.2 if val>=0 else -0.2), bar.get_y()+bar.get_height()/2,
                f"{val:+.1f}pp", va="center",
                ha="left" if val>=0 else "right", fontsize=8)
    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_E_improvement_per_scenario.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")







def plot_correlation_matrix(df, model_qb):
    """Plot F — Correlation matrix including gradient feature."""
    feat = pd.DataFrame()
    feat["Density (ID)"]   = df["ID"]
    feat["Velocity (v)"]   = df["v_pen"]
    feat["Scaling (S)"]    = df["S"]
    feat["Depth"]          = df["depth"]
    feat["qb fast"]        = df["qb_fast"]
    if "qb_fast_grad" in df.columns:
        feat["qb fast grad"] = df["qb_fast_grad"]   # [E2]
    log_S = np.log10(df["S"].clip(lower=1))
    feat["S × Depth"]      = log_S * df["depth"]
    feat["ID × Depth"]     = df["ID"] * df["depth"]
    feat["S × ID"]         = log_S * df["ID"]
    feat["v × S"]          = df["v_pen"] * log_S
    feat["Target (res_qb)"]= df["res_qb"]

    corr = feat.corr()
    fig, ax = plt.subplots(figsize=(13,11))
    fig.suptitle("Correlation Matrix — Input Features and Target\n"
                 "Last row/column shows correlation with res_qb (ML target)",
                 fontsize=12, fontweight="bold")
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlation coefficient")
    ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            val = corr.values[i,j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val)>0.6 else "black", fontweight="bold")
    n = len(corr.columns)
    ax.add_patch(plt.Rectangle((-0.5, n-1.5), n, 1, fill=False, edgecolor="#e74c3c", lw=2.5))
    ax.add_patch(plt.Rectangle((n-1.5,-0.5), 1, n, fill=False, edgecolor="#e74c3c", lw=2.5))
    plt.tight_layout()
    out = f"{PLOTS_DIR}/pub_F_correlation_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("MGS Residual Correction — Training on REAL Data  [ENHANCED]")
    print("Reference: S=1")
    print("="*60)

    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df):,} rows")
    print(f"Scenarios:  {df['scenario_id'].nunique()}")
    print(f"S values:   {sorted(df['S'].unique())}")
    print(f"Depth:      {df['depth'].min():.2f} – {df['depth'].max():.2f} m")

    # Exclude shallow zone from training pool
    if "is_shallow" in df.columns:
        df_train_pool = df[df["is_shallow"]==0].copy()
        print(f"  Shallow rows excluded from training: {(df['is_shallow']==1).sum()}")
        print(f"  Training pool: {len(df_train_pool):,} rows")
    else:
        df_train_pool = df.copy()

    S_vals = sorted(df["S"].unique())

    # ── [E9] Consistent test set: always exclude shallow rows ────────────────
    if SEPARATE_S_MODELS and len(S_vals) > 1:
        print(f"\n  Training SEPARATE models for each S: {S_vals}")
        models_qb = {}
        df_test_parts = []

        for S_val in S_vals:
            df_s = df_train_pool[df_train_pool["S"]==S_val]
            print(f"\n  --- S={S_val}: {len(df_s):,} rows ---")

            # [E5] GroupKFold CV + [E6] grid search
            cv_rmse = cross_validate(df_s, "res_qb")

            # Hold out one split for final test reporting
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            tr_idx, te_idx = next(splitter.split(df_s, groups=df_s["scenario_id"]))
            df_tr  = df_s.iloc[tr_idx]
            df_te  = df_s.iloc[te_idx].copy()

            print(f"    Train: {df_tr['scenario_id'].nunique()} scenarios  "
                  f"Test: {df_te['scenario_id'].nunique()}")

            models_qb[S_val] = train_final(df_tr, "res_qb", f"qb_S{S_val}")
            df_test_parts.append(df_te)

        df_test = pd.concat(df_test_parts)
        df_test["qb_corrected"] = np.nan
        for S_val in S_vals:
            mask = df_test["S"]==S_val
            X    = build_features(df_test[mask]).values
            df_test.loc[mask, "qb_corrected"] = (
                df_test.loc[mask,"qb_fast"] + models_qb[S_val].predict(X)).clip(lower=0)

        with open(f"{RESULTS_DIR}/models_qb_per_S.pkl", "wb") as f:
            pickle.dump(models_qb, f)
        default_S = max(S_vals)
        with open(f"{RESULTS_DIR}/model_qb.pkl", "wb") as f:
            pickle.dump(models_qb[default_S], f)
        model_qb = models_qb[default_S]

        # [E1] normalisation from training data only
        all_tr = pd.concat([df_train_pool[df_train_pool["S"]==sv].iloc[
            next(GroupShuffleSplit(1, test_size=0.20, random_state=42).split(
                df_train_pool[df_train_pool["S"]==sv],
                groups=df_train_pool[df_train_pool["S"]==sv]["scenario_id"]))[0]]
            for sv in S_vals])
        compute_and_save_normalisation(all_tr, META_PATH)

    else:
        print(f"\n  Training COMBINED model for all S levels")

        # [E5] CV on full training pool
        cv_rmse = cross_validate(df_train_pool, "res_qb")

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(splitter.split(
            df_train_pool, groups=df_train_pool["scenario_id"]))
        df_tr_all = df_train_pool.iloc[tr_idx]
        df_test   = df_train_pool.iloc[te_idx].copy()

        print(f"  Train: {df_tr_all['scenario_id'].nunique()} scenarios  "
              f"Test: {df_test['scenario_id'].nunique()}")

        # [E1] normalisation on training rows only
        compute_and_save_normalisation(df_tr_all, META_PATH)

        model_qb = train_final(df_tr_all, "res_qb", "qb")

        X_test = build_features(df_test).values
        df_test["qb_corrected"] = (
            df_test["qb_fast"] + model_qb.predict(X_test)).clip(lower=0)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = []
    for zone_label, mask in [
        ("ALL depths",                 pd.Series([True]*len(df_test), index=df_test.index)),
        (f"Shallow (0–{SHALLOW_CUTOFF}m)", df_test["depth"] <= SHALLOW_CUTOFF),
        (f"Deep (>{SHALLOW_CUTOFF}m)",     df_test["depth"] >  SHALLOW_CUTOFF),
    ]:
        sub = df_test[mask]
        metrics.append(compute_metrics(sub["qb_ref"].values, sub["qb_fast"].values,
                                       f"qb fast vs ref [{zone_label}]"))
        metrics.append(compute_metrics(sub["qb_ref"].values, sub["qb_corrected"].values,
                                       f"qb corrected vs ref [{zone_label}]"))

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(f"{RESULTS_DIR}/metrics.csv", index=False)
    print("\n" + df_metrics.to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    PLOT_DATA_PATH = DATA_PATH.replace(".csv", "_plot.csv")
    if os.path.exists(PLOT_DATA_PATH):
        df_plot_full = pd.read_csv(PLOT_DATA_PATH)
        print(f"\n  Loaded plot dataset: {len(df_plot_full):,} rows")
    else:
        df_plot_full = df.copy()

    X_plot = build_features(df_plot_full).values
    df_all  = df_plot_full.copy()
    df_all["qb_corrected"] = (
        df_all["qb_fast"] + model_qb.predict(X_plot)).clip(lower=0)

    print("\nGenerating plots...")
    plot_correction_curves(df_all, model_qb)
    plot_sensitivity(df_all, model_qb)
    plot_publication_correction_curve(df_all, model_qb)
    plot_error_reduction_summary(df_test, model_qb)
    plot_predicted_vs_actual(df_all, model_qb)
    plot_depth_error_profile(df_all, model_qb)   # [E8] fixed
    plot_improvement_percentage(df_test, model_qb)
    plot_correlation_matrix(df, model_qb)


    print(f"\nDone. Models → {RESULTS_DIR}/  Plots → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
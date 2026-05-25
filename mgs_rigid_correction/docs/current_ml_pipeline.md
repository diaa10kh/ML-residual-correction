# Current ML Residual-Correction Pipeline

This document describes the active ML workflow used by
`scripts/07_build_ml_dataset.py` and `scripts/08_train_correction_model.py`.
It supersedes the older planning notes in `ml_residual_correction_plan.md`.

## Purpose

The ML model corrects high mass-scaling base resistance curves back toward the
`S=1` reference curve. The current model is intentionally for base resistance
`q_b` only. Shaft resistance `q_s` is plotted during raw-data inspection but is
not used as an ML target or ML input feature because it is small, sensitive to
local changes, and there is no matching centrifuge-test `q_s` correction target.

The full-version simulations are generated from the checked Abaqus reference
input. For the CEL setup, the soil mesh/contact definitions are kept unchanged,
while the pile and press coordinates, press displacement, timing, and
material/scaling data are patched per metadata row. The historical folder name
`mgs_rigid_correction` should not be read as a rigid-pile modelling assumption.

For each high-S curve, the target residual is:

```text
res_qb = qb_ref - qb_fast
```

The corrected curve is:

```text
qb_corrected = qb_fast + predicted_res_qb
```

## Dataset Building

The dataset builder reads extracted per-run CSV files from:

```text
data/extracted/per_run_csv/
```

It writes separate datasets for each soil model:

```text
data/processed/ml/mcm/
data/processed/ml/hypoplastic/
```

Each folder contains:

```text
real_dataset.csv
real_dataset_plot.csv
dataset_meta.json
```

The full-version matrix expects 540 CSV files per soil model:

```text
9 geometries x 4 densities x 3 velocities x 5 scaling factors = 540
```

The dataset builder is metadata-driven. It reads the planned geometry and run
information from `run_metadata_full.csv` for hypoplastic runs and
`run_metadata_full_mohr_coulomb.csv` for MCM runs.

The builder pairs each high-S run (`S=10, 30, 50, 100`) with the matching
`S=1` reference using `scenario_id`. It uses exact depth matching, keeps the
same positive-depth domain as the pre-ML raw plots, smooths `q_b` first, then
removes smoothed non-positive `q_b` values. It also computes `qb_fast_grad`,
flags rows shallower than `1 m`, and reports outliers by `S`.

## Training Inputs

The trainer uses only quantities available from the high-S simulation:

```text
ID
v_pen
S
depth
qb_fast
qb_fast_grad
D_m
penetration_m
penetration_over_D
depth_over_D
depth_over_penetration
S_x_depth
ID_x_depth
S_x_ID
v_x_S
grad_x_S
grad_x_ID
S_x_depth_over_D
S_x_penetration_over_D
```

The `S=1` response is not used as an input feature. It is used only to compute
the residual target.

`L_m`, `L_over_D`, `penetration_over_L`, and `depth_over_L` are retained as
metadata/diagnostic values. They are not core ML features in the full-version
model because pile length is linked to the selected penetration depth through
`penetration/L = 0.90`.

The first meter is included in training. The dataset still keeps the
`is_shallow` flag for inspection, but the current trainer uses the complete
depth range and applies the corrected curve from the surface to the maximum
penetration depth.

The trainer uses one residual-correction model, then applies an optional
S-dependent scale factor to the predicted residual. All current factors are
1.0, so no damping is active:

```text
qb_corrected = qb_fast + alpha(S) * predicted_residual
alpha(10) = 1.0
alpha(30) = 1.0
alpha(50) = 1.0
alpha(100) = 1.0
```

The current reported correction is therefore the raw ML residual correction
without additional damping.

## Error Metrics

The primary metric is WAPE:

```text
WAPE(%) = 100 * sum(abs(qb_ref - qb_pred)) / sum(abs(qb_ref))
```

This is used because `q_b` starts near zero and increases nonlinearly with
depth. WAPE avoids the near-zero blow-up that can occur with pointwise
percentage metrics such as MAPE.

The secondary metric is RMSE:

```text
RMSE = sqrt(mean((qb_pred - qb_ref)^2))
```

RMSE is reported in MPa and highlights larger local errors.

The final `metrics.csv` contains full-curve grouped out-of-fold validation rows:

```text
qb fast vs ref
qb corrected vs ref
```

The generated `test_predictions.csv` contains grouped out-of-fold predictions
for all 12 density-velocity scenarios. Each scenario is predicted by a model
trained without that scenario, so all density levels and all velocity levels are
represented in the reported validation metrics:

```text
ID = 0.3, 0.6, 0.8, 0.9
v  = 25, 50, 100 cm/s
```

Additional grouped breakdowns are written to `metrics_by_group.csv`, and the
per-fold scenario metrics are written to `validation_folds.csv`.

For full-version datasets, the trainer also writes stricter robustness checks:

```text
validation_strategy_summary.csv
validation_leave_one_geometry.csv
validation_leave_one_diameter.csv
validation_leave_one_penetration.csv
```

Grouped scenario out-of-fold validation remains the main publication
performance evidence. The geometry, diameter, and penetration holdouts are
robustness checks for stronger extrapolation claims.

`oof_predictions.csv` is the preferred source for validation plots. It is the
same grouped out-of-fold prediction table that is also written as
`test_predictions.csv` for backwards compatibility.

The script no longer reports separate shallow/deep error tables.

## Outputs

Training writes separate results for each soil model:

```text
results/ml/mcm/
results/ml/hypoplastic/
```

Each result folder contains:

```text
model_qb.pkl
metrics.csv
metrics_by_group.csv
oof_predictions.csv
test_predictions.csv
validation_folds.csv
cv_summary.csv
```

Plots are written to:

```text
plots/ml/mcm/
plots/ml/hypoplastic/
```

The active plot set is:

```text
correction_curves_all.png
correction_curves_test_scenarios.png
sensitivity_vs_S.png
sensitivity_vs_ID.png
heatmap_S_vs_ID.png
heatmap_S_vs_depth.png
pub_A_correction_curve_S10.png
pub_A_correction_curve_S30.png
pub_A_correction_curve_S50.png
pub_A_correction_curve_S100.png
pub_B_error_reduction_summary.png
pub_C_predicted_vs_actual.png
pub_D_depth_error_profile.png
pub_E_improvement_per_scenario.png
pub_G_wape_improvement_matrix.png
pub_F_correlation_matrix.png
diagnostic_residual_predicted_vs_actual.png
```

The validation/performance figures are generated from `oof_predictions.csv`:
`correction_curves_test_scenarios.png`, `sensitivity_vs_S.png`,
`sensitivity_vs_ID.png`, both WAPE heatmaps, `pub_A`, `pub_B`, `pub_C`,
`pub_D`, `pub_E`, and `pub_G`. `correction_curves_all.png` uses the final model
on `real_dataset_plot.csv` and is therefore a final-model diagnostic, not
validation evidence. `pub_F_correlation_matrix.png` is feature/target context,
not model validation.

`pub_A_correction_curve_S*.png` selects the best grouped out-of-fold corrected
curve for each S level. It is useful as a clean illustration, but it should be
described as a best-case validation example rather than as a representative
case.

`pub_C_predicted_vs_actual.png` compares actual/reference `q_b` from `S=1`
against the raw high-S `q_b` and the corrected `q_b`. Points on the dashed line
are perfect `q_b` predictions.

`pub_D_depth_error_profile.png` shows mean error versus depth. The depth axis is
plotted with `0 m` at the top and increasing depth downward.

`pub_G_wape_improvement_matrix.png` shows the WAPE improvement in percentage
points for each `(S, ID, v_pen)` combination. Negative cells mark cases where
the correction worsened WAPE.

`diagnostic_residual_predicted_vs_actual.png` compares actual residuals with
model-predicted residuals. It is intentionally stricter and noisier than
`pub_C`; use it as a diagnostic or supplement, not as the main visual summary.

## Commands

Build both soil-model datasets:

```powershell
python scripts\07_build_ml_dataset.py
```

Build one soil-model dataset:

```powershell
python scripts\07_build_ml_dataset.py --soil-model mcm
python scripts\07_build_ml_dataset.py --soil-model hypoplastic
```

Train both soil-model correction models:

```powershell
python scripts\08_train_correction_model.py
```

Train one soil-model correction model:

```powershell
python scripts\08_train_correction_model.py --soil-model mcm
python scripts\08_train_correction_model.py --soil-model hypoplastic
```

`mohr_coulomb` is accepted as a command-line alias for `mcm`.

# Current ML Residual-Correction Pipeline

This document describes the active ML workflow used by
`scripts/07_build_ml_dataset.py` and `scripts/08_train_correction_model.py`.
It supersedes the older planning notes in `ml_residual_correction_plan.md`.

## Purpose

The ML model corrects high mass-scaling base resistance curves back toward the
`S=1` reference curve. The current model is for base resistance `q_b` only.
Shaft resistance `q_s` is plotted during raw-data inspection but is not used as
an ML target or ML input feature.

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

The current Phase 0 matrix expects 60 CSV files per soil model:

```text
4 densities x 3 velocities x 5 scaling factors = 60
```

The builder pairs each high-S run (`S=10, 30, 50, 100`) with the matching
`S=1` reference using `scenario_id`. It uses exact depth matching, filters zero
`qb_MPa` rows before smoothing, computes `qb_fast_grad`, flags rows shallower
than `1 m`, and reports outliers by `S`.

## Training Inputs

The trainer uses only quantities available from the high-S simulation:

```text
ID
v_pen
S
depth
qb_fast
qb_fast_grad
S_x_depth
ID_x_depth
S_x_ID
v_x_S
grad_x_S
grad_x_ID
```

The `S=1` response is not used as an input feature. It is used only to compute
the residual target.

The first meter is excluded from training through the dataset column
`is_shallow`. In practice:

```text
training rows: is_shallow == 0
```

Final evaluation is performed on the complete held-out curves, including the
first meter.

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

The final `metrics.csv` contains only full-curve rows:

```text
qb fast vs ref
qb corrected vs ref
```

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
test_predictions.csv
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
pub_F_correlation_matrix.png
```

`pub_C_predicted_vs_actual.png` compares actual residuals with model-predicted
residuals. Points on the dashed line are perfect residual predictions.

`pub_D_depth_error_profile.png` shows mean error versus depth. The depth axis is
plotted with `0 m` at the top and increasing depth downward.

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

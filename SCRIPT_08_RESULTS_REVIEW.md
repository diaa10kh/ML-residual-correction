# Review of `08_train_correction_model.py` and Generated Results

Date reviewed: 2026-05-24

## Short Judgement

The current script and results are now defensible as an exploratory grouped
out-of-fold validation of an ML residual correction for `q_b`. The numerical
trends are plausible: high mass-scaling factors create larger systematic
errors, and the residual model reduces the integrated base-resistance error in
most cases.

The results are not yet strong enough for an unsupported claim that the model
is generally reliable. The defensible claim is narrower: within this 4 x 3 x 5
simplified deformable-pile matrix, a grouped out-of-fold residual model reduces
`q_b` error overall, especially for larger S, but the hypoplastic S=10 cases
and high-density cases remain weaker.

## Script Logic

The training split is now correct for the current dataset size. The script uses
`GroupKFold` by `scenario_id`, so complete density-velocity scenarios are held
out. Rows from the same scenario are not split between training and validation.
This avoids the main leakage risk from the earlier row-level/random grouped
split.

The validation output covers all expected cases:

- 12 density-velocity scenarios per soil model.
- 4 density levels: `ID = 0.3, 0.6, 0.8, 0.9`.
- 3 velocities: `25, 50, 100 cm/s`.
- 4 corrected high-S levels: `S = 10, 30, 50, 100`.
- 48 corrected curves per soil model.

The saved final model is trained on the full dataset after validation. That is
acceptable for later deployment, provided the paper states that the performance
numbers come from grouped out-of-fold predictions, not from the final fitted
model evaluated on its own training data.

The current alpha values are all 1.0, so no residual damping is applied. The
script keeps raw and applied residual columns, which is useful for later
sensitivity checks.

## Numerical Results

Overall grouped out-of-fold metrics:

| Soil model | WAPE before | WAPE after | RMSE before | RMSE after |
|---|---:|---:|---:|---:|
| MCM | 7.25% | 2.66% | 0.931 MPa | 0.490 MPa |
| Hypoplastic | 6.21% | 3.78% | 1.010 MPa | 0.674 MPa |

By mass-scaling factor, the result is physically plausible. The raw error
increases with S, and the correction reduces it. For MCM, all S levels improve
strongly. For hypoplastic runs, the improvement is clear for S=30, 50, and 100,
but S=10 only improves from 4.57% to 4.11% WAPE. That is a weak correction and
should not be overclaimed.

By scenario, MCM improves in every held-out density-velocity scenario.
Hypoplastic improves overall, but two scenario-level folds are slightly worse:

- `G0_DENS_HIGH_V_LOW`: 4.676% -> 4.816% WAPE.
- `G0_DENS_MED_V_REF`: 3.323% -> 3.351% WAPE.

At the finer scenario-S level, hypoplastic has 6 worsened cases out of 48,
mostly at S=10. This should be shown or stated in the results section.

## Figure Review

`correction_curves_all.png` is a useful diagnostic but not validation evidence.
It uses the final model and the plotting dataset. The title now labels this as
a final-model diagnostic.

`correction_curves_test_scenarios.png` is validation evidence. It uses grouped
out-of-fold predictions and shows the highest-S corrected curve per subplot.
The figure is plausible: the S=100 correction usually moves toward the S=1
reference, but the hypoplastic high-density and high-velocity cases remain less
perfect.

`sensitivity_vs_S.png` is plausible and useful. It supports the statement that
raw MGS error grows with S, while the corrected error is lower. For
hypoplastic, the corrected S=10 point remains high enough that the discussion
should say S=10 is not a strong learning case.

`sensitivity_vs_ID.png` is plausible. High density is the hardest case after
correction, especially for hypoplastic. This matches the visual behaviour in
the correction curves.

`heatmap_S_vs_ID.png` and `heatmap_S_vs_depth.png` are useful diagnostics. They
show that the correction reduces WAPE across most bins, while shallow/high-S
and high-density regions still contain larger residual error. These plots
should be treated as validation summaries, not mechanistic proof.

`pub_A_correction_curve_S*.png` is now more defensible because it uses the
benchmark-like `ID=0.8, v=50 cm/s` case rather than cherry-picking the best
corrected scenario. The curves are plausible and suitable as example figures.

`pub_B_error_reduction_summary.png` supports the headline metric claim. It is a
good main result figure.

`pub_C_predicted_vs_actual.png` now shows actual/reference `q_b` against raw
high-S and corrected `q_b`, which is the clearer main-paper prediction view.
The stricter point-wise residual scatter is retained separately as
`diagnostic_residual_predicted_vs_actual.png`. That diagnostic remains a
caution: for hypoplastic data, residual prediction is weak at S=10 and stronger
for larger S, so the model is better described as reducing integrated curve
error than as accurately predicting every depth-point residual.

`pub_D_depth_error_profile.png` is plausible. It shows mean bias reduction with
depth. MCM is cleaner; hypoplastic still has wider uncertainty bands and local
over/under-correction.

`pub_E_improvement_per_scenario.png` is important and should stay in the
manuscript or supplement. It makes the non-uniform hypoplastic performance
transparent.

`pub_F_correlation_matrix.png` is useful only as feature/target context. It
does not validate the model and should not be used as evidence of causality.

## Publication Use

The corrected, defensible publication claim is:

> A grouped out-of-fold residual model reduces base-resistance error relative
> to high-S simulations within the tested simplified deformable-pile MGS
> matrix, with the strongest benefit at larger S and weaker performance for
> hypoplastic S=10 and some high-density cases.

Avoid claiming:

- that the correction is universally reliable;
- that point-wise residuals are accurately predicted at all depths;
- that the model is validated outside the simulated density, velocity, and S
  domain;
- that `q_s` is corrected or validated.

Before submission, document the grouped out-of-fold protocol, report the
worsened hypoplastic cases, and present the correction as an interpolation
within the simulated matrix rather than a general surrogate for all MGS
conditions.

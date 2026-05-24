# Publication Readiness Review: Mass-Scaling Expansion

## Verdict

The current project is close to a publishable extended study, but it still needs
methodological tightening before it is publication ready.

The original manuscript has a coherent conference-paper story: combine Mass-Gravity-Scaling (MGS) with beam-/shell-volume coupling, validate the approach against the Deeks and White centrifuge benchmark, and report accuracy-efficiency trade-offs for S = 10 and S = 100.

The current project has moved into a different and potentially publishable study: a deformable-pile investigation of the effect of MGS factor, density, and penetration velocity, with a machine-learning residual correction to recover the S = 1 response. That is a valid direction, but it is not yet supported with the level of validation and methodological clarity required for publication.

Recommended decision: revision before submission. The project can become publication ready after the mass-scaling campaign is framed consistently and the ML correction is evaluated more rigorously.

## What I Reviewed

- Initial manuscript: `ICPMG26-manuscript-alkateeb-revised_final.pdf`
- Active workflow documentation:
  - `README.md`
  - `mgs_rigid_correction/README.md`
  - `mgs_rigid_correction/docs/current_ml_pipeline.md`
  - `mgs_rigid_correction/docs/ml_residual_correction_plan.md`
- Matrix/configuration files:
  - `mgs_rigid_correction/configs/matrix_phase0.yaml`
  - `mgs_rigid_correction/configs/matrix_phase0_mohr_coulomb.yaml`
  - `mgs_rigid_correction/configs/matrix_full.yaml`
- Main scripts:
  - `mgs_rigid_correction/scripts/01_generate_matrix.py`
  - `mgs_rigid_correction/scripts/02_generate_inputs_from_reference.py`
  - `mgs_rigid_correction/scripts/05_run_postprocessing.py`
  - `mgs_rigid_correction/scripts/07_build_ml_dataset.py`
  - `mgs_rigid_correction/scripts/08_train_correction_model.py`
- Generated datasets, metrics, and figures under:
  - `mgs_rigid_correction/data/`
  - `mgs_rigid_correction/results/`
  - `mgs_rigid_correction/plots/`

## Main Strengths

The project already has a useful numerical campaign scaffold. The current Phase 0 data include 60 hypoplastic runs and 60 Mohr-Coulomb runs, with 4 density levels, 3 velocity levels, and 5 scaling factors per soil model. The dataset builder now checks that all expected CSV files exist before training, which is good scientific hygiene.

The residual-learning formulation is conceptually clean: use high-S quantities only as inputs and use the S = 1 curve only to define the residual target. This avoids the most obvious leakage error of giving the model the reference response as an input feature.

The current grouped out-of-fold metrics show that the correction can reduce base-resistance error:

| Soil model | Fast WAPE | Corrected WAPE | Fast RMSE | Corrected RMSE |
|---|---:|---:|---:|---:|
| Hypoplastic | 6.21% | 3.78% | 1.01 MPa | 0.67 MPa |
| Mohr-Coulomb | 7.25% | 2.66% | 0.93 MPa | 0.49 MPa |

For hypoplastic runs, the grouped out-of-fold WAPE improves from 8.13% to
4.14% at S = 100. For Mohr-Coulomb runs, it improves from 9.31% to 3.00% at
S = 100. These are promising exploratory results.

The scripts compile successfully.

## Remaining Issues

### 1. The manuscript story and the project story are no longer the same

The ICPMG manuscript is about the combined use of MGS and beam-/shell-volume coupling for efficient pile installation simulations. The current project is a simplified deformable-volume-pile MGS residual-correction study. These are related, but they are not the same contribution.

Do not simply append the current ML correction results to the existing manuscript. That would make the paper unfocused. You need to choose one of two routes:

1. Keep the ICPMG manuscript focused on MGS plus coupling, and mention the simplified deformable-pile ML correction only as future work.
2. Rewrite the paper as a new mass-scaling effects paper, with the original manuscript serving as context and benchmark motivation.

For the expansion you want, the stronger route is a new/restructured paper with a title closer to:

`Effect of Mass-Gravity-Scaling on Deformable-Pile CEL Installation Simulations and Residual Correction to the S = 1 Response`

### 2. The validation metrics must cover all density levels

The current dataset has 12 scenarios per soil model:

```text
4 densities x 3 velocities = 12 scenarios
```

Validation metrics should include all four density levels (`ID=0.3, 0.6, 0.8,
0.9`) and all three velocities (`25, 50, 100 cm/s`). The reported metrics should
remain based on grouped out-of-fold predictions or explicit density/velocity
holdouts, not on one random grouped split.

Required action:

- Report leave-one-scenario-out/grouped out-of-fold or leave-one-density/velocity-out validation.
- Add explicit holdout tests:
  - hold out DENS_LOW
  - hold out DENS_MED
  - hold out V_LOW
  - hold out V_HIGH
  - hold out S = 100 if you want to test extrapolation in scaling factor
- Keep all depth points from the same scenario together; do not split randomly by depth point.
- Report uncertainty bands or fold-to-fold variability, not only one random split.

The current cross-validation standard deviations are large relative to the means:

| Soil model | CV RMSE mean | CV RMSE std | Folds |
|---|---:|---:|---:|
| Hypoplastic | 0.611 MPa | 0.292 MPa | 12 |
| Mohr-Coulomb | 0.389 MPa | 0.307 MPa | 12 |

This indicates that performance is scenario dependent.

### 3. The ML correction keeps an optional alpha hook that must be documented

The code keeps an alpha hook for the predicted residual, but the current alpha values are all `1.0`:

- alpha(10) = 1.0
- alpha(30) = 1.0
- alpha(50) = 1.0
- alpha(100) = 1.0

This is cleaner than a manually damped correction because the reported results are the raw ML residual correction. For publication, state explicitly that no additional damping was applied. Keep both raw and applied residual columns in the output, because they make later sensitivity checks straightforward if damping is reintroduced.

### 4. Target-based outlier removal can bias the reported test performance

The dataset builder removes outliers using the residual target `res_qb = qb_ref - qb_fast` before the train/test split. This uses information from the S = 1 reference and from all scenarios, including later test scenarios.

That is problematic for a publishable ML evaluation.

Required action:

- Avoid target-based row removal before the split, or apply outlier criteria using only training data.
- Prefer physically defined filters, e.g. failed run status, non-convergent extraction, or impossible negative resistance after smoothing.
- Report how many points were removed and why.

The current preprocessing removed:

- 865 hypoplastic rows
- 455 Mohr-Coulomb rows

This is not fatal, but it must be made transparent and methodologically defensible.

### 5. The current MGS implementation needs clearer physical justification

The input-generation script scales soil density and gravity as expected, but it also scales the steel density by S. This may be acceptable depending on how the pile is represented and constrained, but it is not explained in the manuscript/project documentation.

Required action:

- State exactly which densities are scaled: soil only, pile only, or all materials.
- Explain why scaling pile density does or does not affect the prescribed-velocity installation response.
If pile density is scaled, discuss whether pile inertia can enter the prescribed-velocity contact response and why this is acceptable for the intended interpretation.

For an MGS effects paper, this implementation detail cannot remain implicit.

### 6. The Mohr-Coulomb branch is useful for comparison but not yet publication-grade

The Mohr-Coulomb matrix provides a useful contrast to the hypoplastic model, but the parameters appear to be simple density-dependent choices rather than a fully validated calibration against Fraction E sand.

Required action:

- Treat Mohr-Coulomb as a numerical sensitivity comparison, not as a validated physical model, unless you provide calibration evidence.
- Keep the hypoplastic model as the main physical model because it is connected to the original manuscript and the Pucker et al. parameters.

### 7. Figures are not yet publication ready

The generated figures are useful for diagnosis, but many are not ready for a paper. For example, validation plots should avoid internal explanatory text that belongs in notes or captions.

Required action:

- Use concise figure titles or no titles, with full explanations in captions.
- Standardise units: the project mixes `m/s` in configs and `cm/s` in plots.
- Separate diagnostic figures from publication figures.
- For publication, prioritise:
  1. benchmark comparison against centrifuge data,
  2. raw MGS effect versus S,
  3. error versus S and velocity,
  4. corrected versus uncorrected curves on grouped out-of-fold validation scenarios,
  5. runtime/speed-up versus S.

## Recommended Paper Structure for the Expansion

### 1. Introduction

State that the original benchmark showed that moderate MGS can be efficient. The new objective is to quantify the scaling-induced numerical error and test whether a residual correction can recover the S = 1 deformable-pile response.

### 2. Benchmark and Numerical Model

Reuse the original model description, but make the scope explicit:

- simplified deformable volume-element pile only,
- G0 geometry unless the full matrix is run,
- hypoplastic sand as the primary model,
- Mohr-Coulomb only as a secondary comparison if retained.

### 3. Mass-Scaling Campaign

Define the matrix clearly:

- density levels,
- velocity levels,
- S levels,
- penetration depth,
- output variables.

### 4. Raw Effect of Mass Scaling

Before ML, show what MGS does:

- `q_b(z)` for S = 1, 10, 30, 50, 100,
- WAPE/RMSE/IAE versus S,
- sensitivity to density and velocity,
- runtime versus S.

This section is necessary. Without it, the ML correction has no physical context.

### 5. Residual Correction Model

Only after the raw MGS effect is established, introduce the correction model.

Minimum requirements:

- Inputs must be high-S quantities only.
- Targets must be S = 1 residuals.
- Splits must be grouped by scenario.
- Damping/alpha must be fixed before final testing.
- Report raw model prediction and damped correction separately.
- State clearly that only `q_b` is corrected because `q_s` is small, sensitive,
  and not available from the centrifuge benchmark in the same way.

### 6. Validity Limits

This should be a major discussion section, not a minor caveat. The key
publishable insight is not just that ML reduces WAPE, but where the correction
is reliable inside the tested density, velocity, and scaling-factor domain.

### 7. Conclusions

Conclusions should be conservative:

- S = 10 likely remains close to S = 1 but gives limited correction benefit.
- S = 30 to 100 introduces larger systematic errors that can partly be corrected.
- Correctability must be bounded by the training domain.
- The current model is not valid for beam-/shell-volume coupled pile formulations, other pile geometries, or dynamic pile driving unless separately tested.

## Minimum Work Needed Before Submission

1. Decide whether this is a revised ICPMG paper or a new MGS effects paper.
2. Replace the single random test split with scenario-based validation that covers density and velocity extremes.
3. Remove or redesign target-based outlier removal.
4. State explicitly that alpha is currently 1.0 for all S values, i.e. no damping is applied.
5. Add runtime/speed-up results for every S level.
6. Clean the figures into publication-quality plots.
7. State the exact material-density scaling rule, including whether pile density is scaled.
8. Add limitations: simplified deformable volume-element pile only, G0 geometry only unless the full matrix is run, base resistance only.

## Bottom Line

The project is promising, but it is currently an exploratory analysis, not a publication-ready manuscript extension.

The strongest publishable contribution is not simply "ML improves the curves", but:

> High MGS factors introduce systematic, S-, velocity-, and density-dependent resistance errors in deformable-pile CEL installation simulations; these errors can be partly corrected toward the S = 1 response inside a validated training-domain envelope.

That is a clear and defensible contribution, but the current project needs major tightening before it can support it.

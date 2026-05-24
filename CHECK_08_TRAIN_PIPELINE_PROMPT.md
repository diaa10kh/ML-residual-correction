# Prompt For External Review Of `08_train_correction_model.py`

Please review the ML validation and training pipeline in this repository, focusing specifically on:

`mgs_rigid_correction/scripts/08_train_correction_model.py`

Context:

- The simulations use a simplified deformable steel volume-element pile, not a rigid pile.
- The study investigates Mass-Gravity-Scaling correction for CEL pile installation simulations.
- The processed dataset contains 12 density-velocity scenarios per soil model:
  - densities: `ID = 0.3, 0.6, 0.8, 0.9`
  - velocities: `25, 50, 100 cm/s`
  - high scaling factors corrected against S=1: `S = 10, 30, 50, 100`
- The model corrects base resistance `q_b` only.
- Do not request `q_s` training/correction; `q_s` is intentionally excluded because it is small, sensitive, and not available as a matching centrifuge-test correction target.
- Do not focus on the absolute velocity values or `ALLKE/ALLIE`; those are outside this review.

What the script is supposed to do:

1. Load `data/processed/ml/{mcm,hypoplastic}/real_dataset.csv`.
2. Use only high-S information as ML input features:
   - `ID`, `v_pen`, `S`, `depth`, `qb_fast`, `qb_fast_grad`, and interaction features derived from these.
3. Use the S=1 response only to define the target residual:
   - `res_qb = qb_ref - qb_fast`
4. Avoid leakage by grouping all rows from the same `scenario_id` together.
5. Validate using grouped out-of-fold validation:
   - leave out one complete density-velocity scenario,
   - train on the other 11 scenarios,
   - predict the left-out scenario,
   - repeat until all 12 scenarios have out-of-fold predictions.
6. Ensure the final validation outputs include all density levels (`0.3, 0.6, 0.8, 0.9`) and all velocity levels (`25, 50, 100 cm/s`).
7. Train the final saved model on the full dataset after validation, but do not use that final model's predictions as validation metrics.
8. Save:
   - `model_qb.pkl`
   - `metrics.csv`
   - `metrics_by_group.csv`
   - `oof_predictions.csv`
   - `test_predictions.csv` as a backwards-compatible copy of out-of-fold predictions
   - `validation_folds.csv`
   - diagnostic/publication plots.

Questions to answer:

1. Does the script avoid data leakage from S=1 reference curves into ML input features?
2. Does the grouped out-of-fold validation correctly prevent rows from the same `scenario_id` appearing in both training and validation?
3. Do the reported metrics cover all four density levels and all three velocity levels?
4. Is it defensible to train one combined model across all S values, given that S is included as an input feature?
5. The current alpha values are all `1.0`, so the reported correction is the raw ML residual correction without damping. Is this implementation clear, and are raw/applied residual columns still saved consistently?
6. Are `metrics.csv`, `metrics_by_group.csv`, and `validation_folds.csv` sufficient to support publication claims?
7. Are there any remaining methodological weaknesses, especially related to outlier filtering, target leakage, grouped validation, or metric interpretation?

Please give a direct technical review with:

- major issues,
- minor issues,
- recommended code changes,
- and a final judgement on whether this validation design is defensible for a publication.

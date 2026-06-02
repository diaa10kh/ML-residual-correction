# Rigid-pile MGS correction plan

> Historical note: this file is the original planning document. It is kept for
> background only and contains older script names, metrics, and folder ideas.
> The active workflow is documented in `../README.md` and
> `current_ml_pipeline.md`.
>
> Correction note: the active simplified repository now uses a deformable steel
> volume-element pile. The rigid-pile language below is historical planning
> text and should not be used to describe the current simulations.

## Implementation addendum

The workflow has been implemented in `mgs_rigid_correction/` with the following execution split:

1. `.inp` creation is driven only through Abaqus/CAE Python 2.7 by passing one JSON case file to `00_3D_CPT_deek.py`.
2. SLURM submission is handled by `mgs_rigid_correction/scripts/03_submit_slurm_array.sh`; the current template targets Phase 1 (`#SBATCH --array=1-45`) and can be replaced by the cluster-proven script once available.
3. ODB extraction, resampling, comparison, ML training, correction, and plotting are kept as later-stage scripts and should be run only after the `.odb` files have been downloaded or made available locally.
4. The ML stage uses XGBoost if available, scikit-learn histogram gradient boosting as fallback, and a standard-library ridge model as a final dependency-free fallback.

## 1. Main objective

Build a correction workflow for **rigid-pile CEL simulations with high MGS factors**.

The ML model will learn the difference between a high-MGS rigid simulation and the unscaled rigid simulation:

```text
residual_qb(z) = qb_S1_rigid(z) - qb_highS_rigid(z)

residual_qs(z) = qs_S1_rigid(z) - qs_highS_rigid(z)
```

Then:

```text
qb_corrected(z) = qb_highS_rigid(z) + predicted_residual_qb(z)

qs_corrected(z) = qs_highS_rigid(z) + predicted_residual_qs(z)
```

The final study answers:

```text
Can high-MGS rigid-pile penetration simulations be corrected back to the S = 1 rigid response using a simple ML postprocessor?
```

It does **not** claim validity for deformable piles or coupled beam/volume pile models.

## 2. Simulation model

Use only:

| Item               | Choice                                      |
| ------------------ | ------------------------------------------- |
| pile type          | rigid pile                                  |
| soil               | hypoplastic sand with intergranular strain  |
| method             | CEL                                         |
| installation       | displacement-controlled jacking             |
| MGS reference      | S = 1                                       |
| high-MGS cases     | S = 10, 30, 50, 100                         |
| correction target  | S = 1 rigid-pile result                     |
| physical benchmark | Deeks and White centrifuge case for G0 only |

Do not include:

```text
conventional deformable volume pile
beam-volume coupled pile
shell-volume coupled pile
dummy volume contact layer
pile bending output
structural deformation output
```

This is now a pure MGS study.

# 3. Simulation phases

## Phase 0: Reproduce the benchmark case

Purpose: verify that your rigid-pile automation and postprocessing work.

| Phase   | Geometry | Density   | Velocity | S values   | Number |
| ------- | -------- | --------- | -------- | ---------- | -----: |
| Phase 0 | G0       | DENS_HIGH | V_REF    | 1, 10, 100 |      3 |

This should reproduce the rigid-pile curves from your manuscript figure.

Acceptance criteria:

| Check                | Requirement                                             |
| -------------------- | ------------------------------------------------------- |
| qb extraction        | works for all S                                         |
| qs extraction        | works for all S                                         |
| energy extraction    | ALLKE and ALLIE available                               |
| trend                | S = 100 faster than S = 10, S = 10 faster than S = 1    |
| accuracy trend       | S = 100 deviates more than S = 10                       |
| benchmark comparison | G0, DENS_HIGH, V_REF can be compared to centrifuge data |

## Phase 1: Pilot rigid-pile ML study

Use one geometry only, but vary density, velocity, and S.

| Factor           | Levels | Values                  |
| ---------------- | -----: | ----------------------- |
| Geometry         |      1 | G0                      |
| Relative density |      3 | 60%, 75%, 93%           |
| Velocity         |      3 | 0.010, 0.020, 0.040 m/s |
| MGS factor       |      5 | 1, 10, 30, 50, 100      |

Total:

```text
1 × 3 × 3 × 5 = 45 simulations
```

Training pairs:

```text
3 densities × 3 velocities × 4 high-S values = 36 paired cases
```

Purpose:

```text
Check whether the ML correction works before doing the full matrix.
```

## Phase 2: Full rigid-pile matrix

Use geometry, density, velocity, and S.

### Geometry matrix

| Geometry ID | Diameter D [m] |   L/D | Length L [m] | Purpose                        |
| ----------- | -------------: | ----: | -----------: | ------------------------------ |
| G0          |           0.60 | 16.67 |        10.00 | benchmark geometry             |
| G1          |           0.45 | 16.67 |         7.50 | smaller pile, same slenderness |
| G2          |           0.75 | 16.67 |        12.50 | larger pile, same slenderness  |
| G3          |           0.60 | 12.00 |         7.20 | shorter pile                   |
| G4          |           0.60 | 22.00 |        13.20 | longer pile                    |
| G5          |           0.45 | 12.00 |         5.40 | small and short                |
| G6          |           0.45 | 22.00 |         9.90 | small and long                 |
| G7          |           0.75 | 12.00 |         9.00 | large and short                |
| G8          |           0.75 | 22.00 |        16.50 | large and long                 |

### Density matrix

| Density ID | Relative density ID [%] | Purpose               |
| ---------- | ----------------------: | --------------------- |
| DENS_LOW   |                      60 | medium dense          |
| DENS_MED   |                      75 | dense                 |
| DENS_HIGH  |                      93 | benchmark dense state |

Keep the hypoplastic material parameters fixed. Only change the initial state, for example initial void ratio or relative density.

### Velocity matrix

The benchmark velocity is:

```text
v0 = 0.020 m/s
```

| Velocity ID | v/v0 | Velocity [m/s] | Purpose                     |
| ----------- | ---: | -------------: | --------------------------- |
| V_LOW       |  0.5 |          0.010 | more quasi-static           |
| V_REF       |  1.0 |          0.020 | benchmark velocity          |
| V_HIGH      |  2.0 |          0.040 | stronger inertial influence |

### MGS matrix

|   S | Role                     |
| --: | ------------------------ |
|   1 | unscaled rigid reference |
|  10 | moderate MGS             |
|  30 | high MGS                 |
|  50 | high MGS                 |
| 100 | aggressive MGS           |

Total full matrix:

```text
3 diameters x 4 densities x 3 velocities x 5 S values = 180 simulations
```

Number of S = 1 references:

```text
3 x 4 x 3 = 36 reference simulations
```

Number of high-MGS simulations:

```text
3 x 4 x 3 x 4 = 144 high-MGS simulations
```

This is reasonable on the cluster because all simulations are rigid-pile simulations.

# 4. Case naming convention

Use:

```text
run_id = geometry_id + density_id + velocity_id + S
```

Examples:

```text
G0_D045_DENS_HIGH_V_REF_S001
G0_D045_DENS_HIGH_V_REF_S010
G0_D045_DENS_HIGH_V_REF_S030
G0_D045_DENS_HIGH_V_REF_S050
G0_D045_DENS_HIGH_V_REF_S100
```

Use:

```text
scenario_id = geometry_id + density_id + velocity_id
```

Example:

```text
G0_D045_DENS_HIGH_V_REF
```

All runs with the same `scenario_id` are paired together. The only difference between them is S.

# 5. MGS implementation rule

For every S:

```text
rho_scaled = S × rho_original

g_scaled = g_original / S
```

This preserves the initial geostatic stress state:

```text
sigma_v = rho_scaled × g_scaled × z = rho_original × g_original × z
```

Do not change anything else when changing S.

# 6. Required outputs from your postprocessing script

Your existing postprocessing script should produce one CSV file per run.

Required file:

```text
data/extracted/per_run_csv/{run_id}.csv
```

Required columns:

| Column      |       Unit | Description                                 |
| ----------- | ---------: | ------------------------------------------- |
| run_id      |       text | simulation ID                               |
| scenario_id |       text | case ID without S                           |
| S           |     number | mass scaling factor                         |
| z_m         |          m | penetration depth                           |
| qb_MPa      |        MPa | base resistance                             |
| qs_kPa      |        kPa | shaft resistance                            |
| ALLKE       | model unit | kinetic energy                              |
| ALLIE       | model unit | internal energy                             |
| stable_dt   |          s | stable explicit time increment if available |
| time_s      |          s | analysis time                               |
| walltime_h  |          h | runtime if available                        |

Derived columns created later:

```text
eta = z_m / L_m

z_over_D = z_m / D_m

energy_ratio = ALLKE / max(ALLIE, 1e-12)

energy_ratio_cummax = maximum energy_ratio from surface to current depth
```

# 7. Data resampling

All runs must be resampled to the same normalized depth grid.

Use:

```text
eta = z / L
```

Grid:

```text
eta_grid = 0.02 to 0.90 with 200 points
```

Reason:

```text
Different simulations may have slightly different output depths. ML needs aligned curves.
```

For each run, save:

```text
data/processed/resampled_runs/{run_id}.csv
```

# 8. ML dataset construction

For every `scenario_id`:

1. Load the S = 1 curve.
2. Load S = 10, 30, 50, 100 curves.
3. Merge by `eta_grid`.
4. Compute residuals.

Targets:

```text
target_residual_qb_MPa = qb_S1_MPa - qb_highS_MPa

target_residual_qs_kPa = qs_S1_kPa - qs_highS_kPa
```

Each row in the ML dataset is one depth point from one high-MGS run.

With the active reduced matrix:

```text
36 scenarios x 4 high-S values x 200 depth points = 28,800 ML rows
```

# 9. Input features for ML

Use only information from the high-MGS run. Never use S = 1 information as model input.

## Static features

| Feature          | Meaning              |
| ---------------- | -------------------- |
| S                | MGS factor           |
| logS             | log of S             |
| ID_percent       | relative density     |
| D_m              | pile diameter        |
| L_m              | pile length          |
| L_over_D         | pile slenderness     |
| velocity_m_per_s | penetration velocity |
| v_over_v0        | normalized velocity  |
| eta              | z/L                  |
| z_over_D         | z/D                  |
| z_m              | penetration depth    |

## Raw high-MGS response features

| Feature      | Meaning                 |
| ------------ | ----------------------- |
| qb_fast_MPa  | high-S base resistance  |
| qs_fast_kPa  | high-S shaft resistance |
| dqb_deta     | local qb gradient       |
| dqs_deta     | local qs gradient       |
| qb_movmean_5 | local moving mean of qb |
| qs_movmean_5 | local moving mean of qs |
| qb_movstd_5  | local oscillation of qb |
| qs_movstd_5  | local oscillation of qs |
| qb_cummean   | cumulative mean of qb   |
| qs_cummean   | cumulative mean of qs   |

## Dynamic quality features

| Feature              | Meaning                                 |
| -------------------- | --------------------------------------- |
| energy_ratio         | ALLKE/ALLIE at current depth            |
| energy_ratio_cummax  | maximum ALLKE/ALLIE up to current depth |
| stable_dt            | stable explicit time increment          |
| oscillation_index_qb | qb_movstd_5 / qb_movmean_5              |
| oscillation_index_qs | qs_movstd_5 / qs_movmean_5              |

# 10. ML model

Use two separate models:

```text
model_qb: predicts residual_qb

model_qs: predicts residual_qs
```

Recommended model:

```text
XGBoostRegressor
```

Fallback:

```text
HistGradientBoostingRegressor from scikit-learn
```

Recommended first XGBoost settings:

```python
params = {
    "n_estimators": 600,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}
```

# 11. Train/test split

Do **not** split randomly by depth point.

Use grouped splitting:

```text
group = scenario_id
```

This means all depth points and all S values belonging to one scenario stay together.

Use:

```text
GroupKFold with 5 folds
```

Also create special holdout tests:

| Holdout             | Purpose                                      |
| ------------------- | -------------------------------------------- |
| one unseen geometry | tests geometry transfer                      |
| V_HIGH cases        | tests velocity sensitivity                   |
| S = 100 cases       | tests aggressive MGS                         |
| G0_D045_DENS_HIGH_V_REF  | tests benchmark case against centrifuge data |

# 12. Metrics

Report metrics for qb and qs separately.

Main metrics:

| Metric            | Purpose                                      |
| ----------------- | -------------------------------------------- |
| MAE               | average absolute error                       |
| RMSE              | penalizes large local errors                 |
| NRMSE             | normalized comparison                        |
| Bias              | systematic overprediction or underprediction |
| MaxAE             | worst local error                            |
| IAE               | integrated curve error                       |
| improvement ratio | uncorrected RMSE / corrected RMSE            |

Depth zones:

| Zone    | Range              |
| ------- | ------------------ |
| shallow | eta = 0.02 to 0.20 |
| middle  | eta = 0.20 to 0.60 |
| deep    | eta = 0.60 to 0.90 |
| full    | eta = 0.02 to 0.90 |

This is important because the shallow zone will be noisy.

# 13. Validity gate

The ML correction should not hide invalid dynamic behaviour.

Use:

| Class   | Condition                  | Action                           |
| ------- | -------------------------- | -------------------------------- |
| valid   | max ALLKE/ALLIE ≤ 5%       | correction accepted              |
| caution | 5% < max ALLKE/ALLIE ≤ 10% | correction accepted with warning |
| invalid | max ALLKE/ALLIE > 10%      | correction not trusted           |

This should be printed for every corrected run.

Also check whether the run is inside the training range:

| Parameter | Check                         |
| --------- | ----------------------------- |
| S         | inside trained S range        |
| ID        | inside trained ID range       |
| D         | inside trained D range        |
| L/D       | inside trained L/D range      |
| velocity  | inside trained velocity range |

If not:

```text
WARNING: correction is extrapolative.
```

# 14. Folder structure

Use this structure:

```text
mgs_rigid_correction/
  README.md
  configs/
    matrix_phase0.yaml
    matrix_phase1.yaml
    matrix_full.yaml
    training.yaml
  scripts/
    01_generate_matrix.py
    02_generate_abaqus_inputs.py
    03_submit_slurm_array.sh
    04_check_jobs.py
    05_run_postprocessing.py
    06_resample_curves.py
    07_build_ml_dataset.py
    08_train_models.py
    09_evaluate_models.py
    10_correct_new_run.py
    11_make_plots.py
  user_scripts/
    abaqus_model_generator/
    abaqus_postprocessor/
  data/
    raw/
      odb/
      inp/
      sta/
      msg/
      dat/
    extracted/
      per_run_csv/
      run_metadata.csv
    processed/
      resampled_runs/
      paired_dataset.parquet
      train_dataset.parquet
      test_dataset.parquet
    benchmark/
      centrifuge_deeks_white_2006.csv
  models/
    qb_model.pkl
    qs_model.pkl
    feature_columns.json
    training_report.json
  reports/
    figures/
    tables/
    validation_summary.xlsx
  logs/
```

# 15. Interfaces for your existing scripts

Since you already have:

```text
a Python Abaqus model generation script

a Python postprocessing script
```

Cursor should not rewrite them. Cursor should create wrappers around them.

## Model generation wrapper

Expected input:

```python
case_config = {
    "run_id": "G0_D045_DENS_HIGH_V_REF_S100",
    "geometry_id": "G0",
    "D_m": 0.60,
    "L_m": 10.00,
    "L_over_D": 16.67,
    "ID_percent": 93,
    "velocity_m_per_s": 0.020,
    "S": 100,
    "rho_scale": 100,
    "gravity_scale": 0.01,
    "pile_type": "rigid"
}
```

Expected output:

```text
runs/G0_D045_DENS_HIGH_V_REF_S100/G0_D045_DENS_HIGH_V_REF_S100.inp
```

## Postprocessing wrapper

Expected input:

```text
runs/{run_id}/{run_id}.odb
```

Expected output:

```text
data/extracted/per_run_csv/{run_id}.csv
```

Required output columns:

```text
z_m, qb_MPa, qs_kPa, ALLKE, ALLIE, stable_dt, time_s
```

# 16. SLURM workflow

Use SLURM array jobs.

For Phase 1:

```bash
#SBATCH --array=1-45
```

For active reduced matrix:

```bash
#SBATCH --array=1-180
```

Basic structure:

```bash
#!/bin/bash
#SBATCH --job-name=mgs_rigid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5000
#SBATCH --time=50:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

. /etc/profile.d/module.sh
module load abaqus/2023
module load intel/2019

MATRIX_CSV="data/extracted/run_metadata.csv"

RUN_ID=$(awk -F, -v row=$SLURM_ARRAY_TASK_ID 'NR==row+1 {print $1}' $MATRIX_CSV)
INPUT_FILE=$(awk -F, -v row=$SLURM_ARRAY_TASK_ID 'NR==row+1 {print $15}' $MATRIX_CSV)

WORKDIR="runs/${RUN_ID}"
cd "${WORKDIR}"

abaqus job="${RUN_ID}" input="$(basename ${INPUT_FILE})" cpus=${SLURM_CPUS_PER_TASK} interactive
```

You may need to adjust the column number for `INPUT_FILE` depending on the metadata CSV.

# 17. Required plots

## Plot 1: Benchmark comparison

For:

```text
G0_D045_DENS_HIGH_V_REF
```

Show:

```text
centrifuge benchmark
S = 1
S = 10 uncorrected
S = 50 uncorrected
S = 50 corrected
S = 100 uncorrected
S = 100 corrected
```

Separate figures:

```text
qb versus penetration

qs versus penetration
```

## Plot 2: Correction improvement by S

For S = 10, 30, 50, 100:

```text
RMSE uncorrected

RMSE corrected

improvement ratio
```

## Plot 3: Error versus energy ratio

Plot:

```text
x = max ALLKE/ALLIE

y = corrected RMSE
```

This tells you when MGS becomes too dynamic for correction.

## Plot 4: Representative corrected curves

Show cases for:

```text
low density and low velocity
low density and high velocity
high density and low velocity
high density and high velocity
small pile
large pile
short pile
long pile
```

## Plot 5: Feature importance

Show XGBoost feature importance for:

```text
qb model

qs model
```

# 18. Acceptance criteria

## Phase 1 acceptance criteria

| Criterion                   |                                       Target |
| --------------------------- | -------------------------------------------: |
| S = 50 corrected RMSE       |          at least 40% lower than uncorrected |
| S = 100 corrected RMSE      |          at least 40% lower than uncorrected |
| correction worsens cases    |                  less than 10% of test cases |
| benchmark S = 100 corrected |     closer to S = 1 than uncorrected S = 100 |
| energy ratio relation       | high errors correspond to high energy ratios |

## Full matrix acceptance criteria

| Criterion                                    |                                    Target |
| -------------------------------------------- | ----------------------------------------: |
| average improvement ratio                    |                                     > 1.5 |
| deep-zone improvement                        |                           > 2.0 preferred |
| invalid cases detected                       |                                       yes |
| extrapolation warnings                       |                                       yes |
| benchmark corrected high-S versus centrifuge | improved compared with uncorrected high-S |

# 19. Cursor master prompt

Paste this into Cursor:

```text
Implement a Python workflow for a rigid-pile MGS correction study using Abaqus simulation results.

The goal is to isolate Mass-Gravity-Scaling effects only. Therefore, all simulations use a rigid pile. Do not include deformable pile, beam-volume coupling, shell-volume coupling, or dummy volume elements.

The workflow must:
1. Read simulation matrix settings from YAML.
2. Generate run_metadata.csv with one row per Abaqus run.
3. Call my existing Abaqus model generation script through a wrapper.
4. The wrapper must pass geometry, density, velocity, S, rho_scale, gravity_scale, run_id, and pile_type = "rigid".
5. Submit the simulations through a SLURM array.
6. Call my existing Abaqus postprocessing script through a wrapper.
7. The postprocessing output must contain z_m, qb_MPa, qs_kPa, ALLKE, ALLIE, stable_dt, and time_s.
8. Resample all curves to a common eta = z/L grid with 200 points from eta = 0.02 to eta = 0.90.
9. Pair each high-MGS run with the corresponding S = 1 rigid reference using scenario_id.
10. Build an ML dataset where all inputs come only from the high-MGS run.
11. The targets are:
    target_residual_qb = qb_S1 - qb_highS
    target_residual_qs = qs_S1 - qs_highS
12. Train two models:
    one model for qb residual correction,
    one model for qs residual correction.
13. Use XGBoostRegressor if available. Otherwise use HistGradientBoostingRegressor.
14. Use grouped cross validation by scenario_id. Never split randomly by depth point.
15. Evaluate uncorrected and corrected curves using MAE, RMSE, NRMSE, Bias, MaxAE, IAE, and improvement ratio.
16. Report metrics separately for shallow, middle, deep, and full depth zones.
17. Implement a validity gate based on maximum ALLKE/ALLIE:
    valid if <= 5%,
    caution if > 5% and <= 10%,
    invalid if > 10%.
18. Produce plots comparing S = 1, high-S uncorrected, high-S corrected, S = 10, and the centrifuge benchmark for the original benchmark case only.
19. Save trained models, feature columns, metric tables, corrected curves, and figures.

Start with Phase 1:
G0 only,
ID = 60, 75, 93%,
velocity = 0.010, 0.020, 0.040 m/s,
S = 1, 10, 30, 50, 100,
total 45 simulations.

Then support the active reduced matrix:
3 diameter-only geometries,
4 densities,
3 velocities,
5 S values,
total 180 simulations.

The code must be modular and robust to failed Abaqus runs. It must not use any S = 1 quantities as ML input features.
```

# 20. Final study statement

The final study should be framed like this:

```text
This study investigates whether the systematic error introduced by high Mass-Gravity-Scaling in rigid-pile CEL simulations can be learned and corrected using a postprocessing model. The correction is trained against unscaled rigid-pile simulations and assessed with respect to numerical accuracy, energy-based validity limits, and the centrifuge benchmark for the original reference case.
```

That is clean, defensible, and fully aligned with your goal of isolating MGS effects only.

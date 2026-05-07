# Rigid-Pile MGS Residual Correction Workflow

This folder implements the rigid-pile Mass-Gravity-Scaling workflow described in
`../ml_residual_correction_plan.md`, with the benchmark assumptions checked
against `../ICPMG26-manuscript-alkateeb-revised_final.pdf`.

The current execution order is staged:

1. Generate the simulation matrix and one `case_config.json` per run.
2. Use Abaqus/CAE Python 2.7 to write one `.inp` file per run.
3. Submit the generated `.inp` files on the cluster through a SLURM array.
4. After the `.odb` files are available locally, run postprocessing, resampling,
   pairing, model training, evaluation, correction, and plots.

## Scientific Message

The main argument of this project is:

```text
Mass-Gravity-Scaling accelerates large-deformation geotechnical simulations,
but introduces systematic response errors. These errors are learnable from a
limited simulation matrix and can be corrected using a residual ML model,
enabling faster simulations while preserving reference-quality
force-penetration behavior.
```

The scientific contribution should not be framed as only:

```text
we trained ML
```

The stronger paper message is:

```text
we made accelerated Abaqus simulations more reliable and quantified the
accuracy-speed tradeoff.
```

In practical terms, the trained residual model is applied as:

```text
q_corrected(eta) = q_scaled(eta) + residual_ML(eta, S, density, velocity, ...)
```

where `q_scaled` is the raw high-MGS Abaqus response, `residual_ML` is the
learned correction, and `q_corrected` should approach the expensive `S=1`
reference response.

## Proposed Paper Plots

The final paper should include plots that show the problem, the correction, and
the computational benefit.

1. Workflow diagram

   ```text
   Abaqus MGS simulations -> ODB extraction -> force-depth curves ->
   residual calculation -> ML training -> corrected response
   ```

2. Raw MGS error curves

   Plot `q` or pile reaction force versus normalized penetration `eta` for
   `S=1`, `S=10`, `S=30`, `S=50`, and `S=100`. This shows how the scaled
   simulations deviate from the reference.

3. Before and after correction curves

   Plot the `S=1` reference, the raw high-S result, and the corrected high-S
   result on the same axes. This is the main visual proof that the correction
   works.

4. Residual curves

   Plot:

   ```text
   residual(eta) = q_reference(eta) - q_scaled(eta)
   ```

   for different MGS factors, densities, velocities, and soil models. This
   shows the pattern that the ML model is learning.

5. Error reduction plot

   Plot RMSE, MAE, or normalized error before and after correction for each MGS
   factor. This gives a compact quantitative comparison.

6. Parity plot

   Plot corrected response versus `S=1` reference response with a 1:1 line.
   Good correction should cluster near the 1:1 line.

7. Speedup versus accuracy tradeoff

   Plot computational cost or speedup against error for raw and corrected MGS
   simulations. This figure answers the engineering question: how much faster
   can the simulation be while keeping acceptable accuracy?

8. Generalization test

   Hold out one density, velocity, or soil model from training and test whether
   the correction still works. This helps show that the method is not only
   memorizing the simulation matrix.

9. Feature importance or SHAP plot

   If the residual model is based on XGBoost or another tree model, plot which
   inputs control the correction most strongly, such as `eta`, `S`, velocity,
   density, void ratio, and raw response.

## Run Naming

Each simulation is identified by one `run_id`:

```text
run_id = geometry_id + density_id + velocity_id + S
```

The exact format is:

```text
{geometry_id}_{density_id}_{velocity_id}_S{S as 3 digits}
```

Examples:

```text
G0_DENS_REF_V_REF_S001
G0_DENS_REF_V_REF_S010
G0_DENS_REF_V_REF_S030
G0_DENS_REF_V_REF_S050
G0_DENS_REF_V_REF_S100
```

The `scenario_id` is the same name without the MGS factor:

```text
scenario_id = geometry_id + density_id + velocity_id
```

Example:

```text
scenario_id = G0_DENS_REF_V_REF
```

All runs with the same `scenario_id` belong together. The only difference
between them is the MGS factor `S`. This is how the workflow pairs each high-S
run with its corresponding S=1 reference.

Generated Abaqus files use the `run_id` directly:

```text
runs/{run_id}/{run_id}.inp
runs/{run_id}/{run_id}.odb
runs/{run_id}/{run_id}.sta
runs/{run_id}/{run_id}.msg
```

For example:

```text
runs/G0_DENS_REF_V_REF_S100/G0_DENS_REF_V_REF_S100.inp
runs/G0_DENS_REF_V_REF_S100/G0_DENS_REF_V_REF_S100.odb
```

## Void Ratio

For the hypoplastic model, the first solution-dependent state variable in the
inserted Abaqus keyword block is the initial void ratio. It is no longer
hard-coded in `02_Create_3D_CPT_deek_Voll.py`; it is written per run through
`case_config.json`.

In Phase 0 and Phase 1, every `DENS_REF` run keeps the experimental
two-layer profile:

```text
DENS_REF
Soil_Void        e = 0.615854
Soil_Upper_Layer e = 0.615854
Soil_down_Layer  e = 0.55
```

The active Phase 0 and Phase 1 density IDs are:

```text
DENS_MED  = ID 0.6
DENS_REF  = experimental hard-coded void-ratio profile
DENS_HIGH = ID 0.9
```

The active Phase 0 and Phase 1 velocity IDs are:

```text
V_LOW  = 0.25 m/s
V_REF  = 0.50 m/s
V_HIGH = 1.00 m/s
```

All non-`DENS_REF` scenarios use one uniform void ratio for all three sets:

```text
Soil_Void = Soil_Upper_Layer = Soil_down_Layer = e(ID)
```

The uniform value is computed from `ID_percent` using:

```text
D = (e_max - e) / (e_max - e_min) * (1 + e_min) / (1 + e)
```

with the current defaults:

```text
e_min = 0.49
e_max = 0.76
```

These settings are in the matrix config files under `void_ratio`. The generated
metadata and each `case_config.json` contain:

```text
void_ratio_mode
use_reference_void_profile
void_ratio_soil_void
void_ratio_upper_layer
void_ratio_down_layer
```

## Phase 1 Quick Start

Run from this folder:

```powershell
python scripts\01_generate_matrix.py --config configs\matrix_phase1.yaml --soil-model Hypoplastisch
python scripts\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata.csv --soil-model Hypoplastisch --overwrite-config
```

Choose the constitutive model at the first command with `--soil-model`:

```powershell
python scripts\01_generate_matrix.py --config configs\matrix_phase1.yaml --soil-model Hypoplastisch
python scripts\01_generate_matrix.py --config configs\matrix_phase1.yaml --soil-model Mohr-Coulomb
```

You can also override it when writing the Abaqus case configs:

```powershell
python scripts\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata.csv --soil-model Hypoplastisch --overwrite-config
python scripts\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata.csv --soil-model Mohr-Coulomb --overwrite-config
```

The second command prepares `runs\{run_id}\case_config.json` and a
`generate_input.ps1` helper for each run. To make Abaqus actually write `.inp`
files, add `--execute`:

```powershell
python scripts\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata.csv --soil-model Hypoplastisch --overwrite-config --execute
```

For a one-case smoke test:

```powershell
python scripts\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata.csv --soil-model Hypoplastisch --overwrite-config --execute --run-id G0_DENS_REF_V_REF_S001
```

Then check generated files:

```powershell
python scripts\04_check_jobs.py
```

Expected after the full Phase 1 input generation:

```text
inp=45
```

Expected after the full Phase 0 input generation:

```text
inp=27
```

The wrapper passes each case to Abaqus through the `MGS_CASE_CONFIG`
environment variable and then calls:

```text
abaqus cae noGUI=..\00_3D_CPT_deek.py
```

Do not open or run `..\00_3D_CPT_deek.py` directly in Abaqus/CAE for this
workflow. Without `MGS_CASE_CONFIG`, it does not know which run folder and
configuration to use.

The Abaqus generator path is intentionally Python 2.7-compatible. The
postprocessing and ML scripts can be run with a normal Python environment.

## Cluster Submission

The SLURM template is:

```text
scripts/03_submit_slurm_array.sh
```

It follows the working TUHH cluster pattern from your earlier project: upload
all `.inp` files into one flat remote directory, submit an oversized guarded
array, copy each selected input to a job-specific `/work/gbt/...` directory,
run Abaqus there, and copy result files back to the submit directory.

The local WSL helper is:

```text
scripts/12_cluster_transfer_submit_fetch.sh
```

Submit after the `.inp` files exist:

```bash
./scripts/12_cluster_transfer_submit_fetch.sh submit_all 0 0
```

Fetch results later:

```bash
./scripts/12_cluster_transfer_submit_fetch.sh fetch 0 0
```

On Windows, `scripts/00_submit_mgs_jobs.cmd` calls the WSL helper and then opens
the cluster terminal. Set `MGS_CLUSTER_PASSWORD` first if you do not want to edit
the script.

## ODB Stage

After downloading the `.odb` files into `runs/{run_id}/`, the intended order is:

```powershell
python scripts\05_run_postprocessing.py --execute
python scripts\06_resample_curves.py
python scripts\07_build_ml_dataset.py
python scripts\08_train_models.py
python scripts\09_evaluate_models.py
python scripts\11_make_plots.py
```

`04_postprocessing.py` is the Abaqus-side ODB extractor. It writes one CSV per
run to:

```text
data/extracted/per_run_csv/{run_id}.csv
```

Required columns are:

```text
run_id, scenario_id, S, z_m, qb_MPa, qs_kPa, ALLKE, ALLIE, stable_dt, time_s, walltime_h
```

## Important Modelling Rules

The `.inp` generation wrapper uses only rigid piles. For each run:

```text
rho_scaled = S * rho_original
g_scaled = g_original / S
```

The ML inputs are built only from the high-S run. S=1 curves are used only to
build the residual targets and evaluation reference.

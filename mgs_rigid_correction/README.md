# Deformable-Pile MGS Residual Correction Workflow

This folder is the active workflow. The simplified Phase 0 path generates
Abaqus input files from one checked reference input file, while the full-version
geometry path prepares Abaqus/CAE model-building cases so pile diameter and
model dimensions can change. Both paths then run simulations on the cluster,
extract ODB curves, and train a residual correction model.

Important modelling note: despite the historical folder name
`mgs_rigid_correction`, the active simplified simulations use a deformable steel
volume-element pile. The checked reference input contains a steel `C3D8R` pile
with a solid section assigned to material `Stahl`; the rigid-body constraint in
the reference input applies to the press/loading body (`Presse`), not to the
pile itself.

Run commands from this folder unless noted otherwise:

```powershell
cd mgs_rigid_correction
```

On Windows, if `python` opens the Microsoft Store instead of the project
environment, use the repository virtual environment explicitly:

```powershell
..\.venv\Scripts\python.exe -m pip install -r ..\requirements.txt
```

## Active Files

```text
configs/                              matrix definitions
reference_inputs/CPT_90_MCM_...inp    checked Mohr-Coulomb reference input
abaqus/vumat-hypo-2020-hst.for        VUMAT for hypoplastic runs
abaqus/full_geometry_generator/       full-version Abaqus model builder
scripts/01_generate_matrix.py         write run metadata
scripts/02_generate_inputs_from_reference.py
scripts/03_submit_slurm_array.sh      remote SLURM array script
scripts/04_check_jobs.py              check expected files
scripts/05_run_postprocessing.py      prepare/run ODB extraction
scripts/06_plot_pre_ml_raw_curves.py  plot raw extracted curves before ML
scripts/07_build_ml_dataset.py        build paired residual datasets
scripts/08_train_correction_model.py  train model and plots
```

The Abaqus/CAE model-building generator is not used for the simplified Phase 0
reference-input workflow. The `full-version` geometry workflow uses it because
the reference `.inp` patcher cannot change pile diameter and model dimensions.
It is kept under:

```text
abaqus/full_geometry_generator/
```

## Quick Start: Full-Version Geometry Metadata

The full-version matrix plans 540 runs per soil model:

```text
9 geometries x 4 densities x 3 velocities x 5 scaling factors = 540
```

Generate hypoplastic metadata only:

```powershell
python scripts\01_generate_matrix.py --soil-model hypoplastic
```

Generate MCM metadata only:

```powershell
python scripts\01_generate_matrix.py --soil-model mcm
```

Generate both metadata files:

```powershell
python scripts\01_generate_matrix.py --soil-model both
```

The full-version metadata files are:

```text
data/extracted/run_metadata_full.csv
data/extracted/run_metadata_full_mohr_coulomb.csv
```

Generate the 540 hypoplastic `.inp` files from the checked reference input:

```powershell
python scripts\02_generate_inputs_from_reference.py --metadata data\extracted\run_metadata_full.csv --soil-model Hypoplastisch --clean
```

The full-version run names include the geometry dimensions. For example,
`G5_D060_P110_DENS_HIGH_V_LOW_S001` means geometry `G5`, `D=0.60 m`,
`penetration=11.0 m`, high density, low velocity, and `S=1`.

The full-version geometry design varies pile diameter and capped penetration
depth. Pile length is linked to penetration through `penetration/L = 0.90`, so
`L_m`, `L_over_D`, and `penetration_over_L` are metadata/diagnostic values, not
independent geometry effects.

The direct reference-input patcher changes only the pile and press coordinates,
press displacement, timing, and material/scaling data. The CEL soil mesh and
contact definitions are copied unchanged.

## Quick Start: Phase 0 Hypoplastic

Generate the 60-run Phase 0 matrix:

```powershell
python scripts\01_generate_matrix.py --config configs\matrix_phase0.yaml --output data\extracted\run_metadata_phase0.csv
```

Create `.inp` files by patching the reference input:

```powershell
python scripts\02_generate_inputs_from_reference.py --metadata data\extracted\run_metadata_phase0.csv --soil-model Hypoplastisch --clean
```

`--clean` only removes previously generated input/helper files in the selected
run folders. It does not remove `.odb`, `.sta`, `.msg`, `.dat`, or output logs.

Check generated files:

```powershell
python scripts\04_check_jobs.py --metadata data\extracted\run_metadata_phase0.csv --output reports\tables\job_status_phase0.csv
```

Expected full Phase 0 hypoplastic input count:

```text
inp=60
```

## Quick Start: Phase 0 Mohr-Coulomb

Generate the Mohr-Coulomb matrix:

```powershell
python scripts\01_generate_matrix.py --config configs\matrix_phase0_mohr_coulomb.yaml --output data\extracted\run_metadata_phase0_mohr_coulomb.csv
```

Create the 60 Mohr-Coulomb `.inp` files:

```powershell
python scripts\02_generate_inputs_from_reference.py --metadata data\extracted\run_metadata_phase0_mohr_coulomb.csv --soil-model Mohr-Coulomb --clean
```

Check generated files:

```powershell
python scripts\04_check_jobs.py --metadata data\extracted\run_metadata_phase0_mohr_coulomb.csv --output reports\tables\job_status_phase0_mohr_coulomb.csv
```

Expected full Phase 0 Mohr-Coulomb input count:

```text
inp=60
```

## Optional Phase 1

Phase 1 remains available through `configs/matrix_phase1.yaml`. It uses the
same reference-input generator:

```powershell
python scripts\01_generate_matrix.py --config configs\matrix_phase1.yaml --output data\extracted\run_metadata.csv
python scripts\02_generate_inputs_from_reference.py --metadata data\extracted\run_metadata.csv --soil-model Hypoplastisch --clean
```

Expected full Phase 1 input count:

```text
inp=60
```

## Cluster Run

After `.inp` files exist, submit from WSL:

```bash
bash scripts/12_cluster_transfer_submit_fetch.sh submit_all
```

On the full-version branch this submits the 540 hypoplastic inputs listed in:

```text
data/extracted/run_metadata_full.csv
```

to the default remote folder:

```text
/work/gbt/$USER/MGS_Rigid_Correction/FULL_HYPO
```

By default the helper submits the full job array without an artificial task
limit. To add a throttle intentionally, set `MGS_ARRAY_THROTTLE`, for example:

```bash
MGS_ARRAY_THROTTLE=40 bash scripts/12_cluster_transfer_submit_fetch.sh submit_all
```

Useful overrides:

```bash
MGS_MATRIX_CSV="data/extracted/run_metadata_phase0_mohr_coulomb.csv" \
MGS_ROOT_REMOTE="/work/gbt/cda6556/MGS_Rigid_Correction/PHASE0_MC" \
bash scripts/12_cluster_transfer_submit_fetch.sh submit_all
```

Fetch results later:

```bash
bash scripts/12_cluster_transfer_submit_fetch.sh fetch
```

If the Codex app cannot see your WSL distribution but your normal terminal can,
start the controlled bridge from a normal PowerShell terminal:

```powershell
.\scripts\14_wsl_cluster_bridge.ps1
```

The bridge only accepts these fixed actions through
`.codex_cluster_bridge/request.json`: `status`, `cancel_all`, `cancel_job`,
`submit_hypo`, `submit_mc`, `fetch_hypo`, `fetch_mc`, and `exit`.

The SLURM script checks each input. If it contains `*User Material`, Abaqus is
run with `vumat-hypo-2020-hst.for`; otherwise it runs without a user routine.

## Postprocessing and ML

After `.odb` files are back in `runs/{run_id}/`:

```powershell
python scripts\05_run_postprocessing.py --metadata data\extracted\run_metadata_phase0.csv --execute
python scripts\06_plot_pre_ml_raw_curves.py
python scripts\07_build_ml_dataset.py
python scripts\08_train_correction_model.py
```

`06_plot_pre_ml_raw_curves.py` is a pre-ML inspection step. It reads extracted
CSV files directly from `data/extracted/per_run_csv/`, applies only a moving
average, and writes raw curve PNGs under:

```text
plots/raw_averaged_plots/{mcm,hypoplastic}/
```

The moving-average and `q_s` axis limits are set near the top of the script.
The current defaults are 20 points for MCM and 30 points for hypoplastic.

`07_build_ml_dataset.py` validates the extracted CSV files against the metadata
matrix. On the full-version branch it expects the metadata-driven full matrix
by default: 540 extracted CSVs per soil model. It stops if files are missing,
so stale partial results are not trained by accident. For an exploratory
partial-data build only, add:

```powershell
python scripts\07_build_ml_dataset.py --allow-partial
```

Both ML scripts also accept `--soil-model mcm`, `--soil-model hypoplastic`, or
`--soil-model both`. The name `mohr_coulomb` is accepted as an alias for `mcm`.

`07_build_ml_dataset.py` expects per-run CSV files in:

```text
data/extracted/per_run_csv/
```

It writes model-specific datasets under:

```text
data/processed/ml/mcm/
data/processed/ml/hypoplastic/
```

`08_train_correction_model.py` writes generated outputs under:

```text
results/ml/mcm/
results/ml/hypoplastic/
plots/ml/mcm/
plots/ml/hypoplastic/
```

Each result folder contains the final model, overall validation metrics,
grouped validation metrics (`metrics_by_group.csv`), out-of-fold predictions
(`oof_predictions.csv`; also written as `test_predictions.csv` for backwards
compatibility), and per-scenario validation fold summaries
(`validation_folds.csv`).

These generated folders are ignored by Git.

The ML model is trained only for base resistance `q_b`. It predicts the
residual correction:

```text
res_qb = qb_ref - qb_fast
qb_corrected = qb_fast + predicted_res_qb
```

Shaft resistance `q_s` is intentionally not trained or corrected in this
workflow. It is small, sensitive to local changes, and there is no matching
centrifuge-test `q_s` result available for the same correction target.

Only high-S quantities are used as ML input features. The S=1 response is used
only as the reference target. The current model is trained on the complete
depth range, including rows flagged as shallow by the dataset builder
(`depth < 1 m`), and the green ML correction curve is plotted from the surface.
The predicted residual is applied through an optional S-dependent scale factor.
All current factors are set to 1.0, so no damping is active:

```text
alpha(S=10) = 1.0
alpha(S=30) = 1.0
alpha(S=50) = 1.0
alpha(S=100) = 1.0
```

Thus, the current reported correction is the raw ML residual correction without
additional damping.

The final reported metrics are:

```text
WAPE(%) = 100 * sum(abs(qb_ref - qb_pred)) / sum(abs(qb_ref))
RMSE    = sqrt(mean((qb_pred - qb_ref)^2))
```

The generated `metrics.csv` and `test_predictions.csv` are based on grouped
out-of-fold validation by `scenario_id`. Each density-velocity scenario is
predicted by a model trained without that scenario, so the reported validation
metrics cover all 12 density-velocity scenarios. The processed dataset contains
all four density levels (`ID=0.3, 0.6, 0.8, 0.9`) and all three velocity levels
(`25, 50, 100 cm/s`):

```text
4 densities x 3 velocities x 4 high-S correction cases = 48 corrected curves
```

Additional grouped breakdowns are written to `metrics_by_group.csv`, and the
per-fold scenario metrics are written to `validation_folds.csv`.

The generated plot set includes correction curves, WAPE sensitivity plots,
WAPE heatmaps, error-reduction summaries, predicted-vs-actual `q_b` plots,
depth error profiles, scenario-improvement bars, a WAPE improvement matrix, a
residual-prediction diagnostic plot, and a feature/target correlation matrix.
Most publication-style performance figures are generated from
`oof_predictions.csv`, so they use grouped out-of-fold predictions. The
`correction_curves_all.png` panel is explicitly a final-model diagnostic, not
validation evidence.

`pub_A_correction_curve_S*.png` is a best-case grouped out-of-fold example for
each S level. Use it as an illustrative correction curve and pair it with
`pub_B`, `pub_E`, or `pub_G` when discussing validation strength. `pub_C` now
plots actual/reference `q_b` against raw and corrected predicted `q_b`; the
older residual-vs-residual view is saved as
`diagnostic_residual_predicted_vs_actual.png`.

More detail is documented in:

```text
docs/current_ml_pipeline.md
```

## Run Naming

Each simulation uses:

```text
{geometry_id}_{density_id}_{velocity_id}_S{S as 3 digits}
```

Examples:

```text
G0_DENS_REF_V_REF_S001
G0_DENS_REF_V_REF_S010
G0_DENS_REF_V_REF_S100
```

Mohr-Coulomb runs are prefixed with `MC_`:

```text
MC_G0_DENS_REF_V_REF_S001
```

All runs with the same `scenario_id` belong together. High-S runs are paired
with the corresponding `S=1` reference for residual learning.

## Modelling Rules

For every generated input:

```text
rho_scaled = S * rho_original
g_scaled   = g_original / S
```

The active input generator applies the density scaling to the soil material and
to the deformable steel pile material. The gravity scaling is applied to the
soil gravity load. The pile remains a deformable volume-element pile; it is not
converted into a rigid body by the simplified workflow.

The full-version matrix uses nine diameter/penetration geometries, four
relative-density levels (`ID=0.3, 0.6, 0.8, 0.9`), three penetration velocities
(`0.25, 0.50, 1.00 m/s`, equivalent to `25, 50, 100 cm/s`), and five scaling
factors (`1, 10, 30, 50, 100`). That is
`9 x 4 x 3 x 5 = 540` simulations per soil model.

Field output is intentionally sparse to reduce ODB size. The generated
`Einpressen` step uses:

```text
field_interval = step_time / 9
```

That gives nine field-output intervals:

```text
9 s step  -> *Output, field, time interval=1
36 s step -> *Output, field, time interval=4
```

The history output interval is not changed from the previous high-frequency
setting.

Hypoplastic runs keep the reference two-layer void-ratio profile for all
`DENS_REF` cases and use uniform void ratios for other densities. Mohr-Coulomb
input files contain `*Elastic`, `*Mohr Coulomb`, and `*Mohr Coulomb Hardening`.
Hypoplastic input files contain `*User Material`, `*Depvar`, and solution
initial conditions.

## Local Checks

Syntax check:

```powershell
python -B -c "import pathlib; [compile(p.read_text(encoding='utf-8-sig'), str(p), 'exec') for p in pathlib.Path('scripts').glob('*.py')]"
```

Smoke-test one input file:

```powershell
python scripts\02_generate_inputs_from_reference.py --metadata data\extracted\run_metadata_phase0.csv --run-id G0_DENS_REF_V_REF_S001 --clean
```

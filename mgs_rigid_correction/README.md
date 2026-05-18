# Rigid-Pile MGS Residual Correction Workflow

This folder is the active workflow. It generates Abaqus input files from one
checked reference input file, runs the simulations on the cluster, extracts ODB
curves, and trains a residual correction model.

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
scripts/01_generate_matrix.py         write run metadata
scripts/02_generate_inputs_from_reference.py
scripts/03_submit_slurm_array.sh      remote SLURM array script
scripts/04_check_jobs.py              check expected files
scripts/05_run_postprocessing.py      prepare/run ODB extraction
scripts/06_resample_curves.py         resample extracted curves
scripts/07_build_ml_dataset.py        build paired residual datasets
scripts/08_train_correction_model.py  train model and plots
```

The old Abaqus/CAE model-building generator is not active. It is kept for later
under:

```text
on_hold/complex_abaqus_generator/
```

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

The SLURM script checks each input. If it contains `*User Material`, Abaqus is
run with `vumat-hypo-2020-hst.for`; otherwise it runs without a user routine.

## Postprocessing and ML

After `.odb` files are back in `runs/{run_id}/`:

```powershell
python scripts\05_run_postprocessing.py --metadata data\extracted\run_metadata_phase0.csv --execute
python scripts\06_resample_curves.py --metadata data\extracted\run_metadata_phase0.csv --matrix-config configs\matrix_phase0.yaml
python scripts\07_build_ml_dataset.py
python scripts\08_train_correction_model.py
```

`07_build_ml_dataset.py` validates the extracted CSV files against the metadata
matrix. For the current Phase 0 matrix it expects 60 extracted CSVs per soil
model. It stops if files are missing, so stale partial results are not trained
by accident. For an exploratory partial-data build only, add:

```powershell
python scripts\07_build_ml_dataset.py --allow-partial
```

Both ML scripts also accept `--soil-model mohr_coulomb` or
`--soil-model hypoplastic` when only one branch should be rebuilt. The older
short name `mcm` is still accepted as an alias for `mohr_coulomb`.

`07_build_ml_dataset.py` expects per-run CSV files in:

```text
data/extracted/per_run_csv/
```

It writes model-specific datasets under:

```text
data/processed/ml/
```

`08_train_correction_model.py` writes generated outputs under:

```text
results/ml/
plots/ml/
```

These generated folders are ignored by Git.

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

The matrix currently uses four relative-density levels (`ID=0.3, 0.6, 0.8,
0.9`), three penetration velocities (`0.25, 0.50, 1.00 m/s`, equivalent to
`25, 50, 100 cm/s`), and five scaling factors (`1, 10, 30, 50, 100`). That is
`4 x 3 x 5 = 60` simulations per geometry and soil model.

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

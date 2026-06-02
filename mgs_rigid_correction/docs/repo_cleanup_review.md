# Repository Cleanup Review

This review records the active full-version workflow state and the files or
folders that are intentionally outside the committed source tree.

## Active Source Scope

The active workflow is the diameter-only full-version matrix:

```text
3 diameters x 4 densities x 3 velocities x 5 scaling factors = 180 runs per soil model
```

The active scaling factors are:

```text
S = 7, 15, 30, 50, 100
```

`S=7` is the reference curve used for residual learning. `S=15`, `S=30`,
`S=50`, and `S=100` are corrected back toward that reference.

The active checked reference input is:

```text
reference_inputs/Pile_11_m_einpressen_Voll_S001.inp
```

The filename is historical. The active metadata and generated run IDs use
`S007` for the reference simulations.

## Active Configs

Only these matrix configs are active:

```text
configs/matrix_full_hypoplastic.yaml
configs/matrix_full_mohr_coulomb.yaml
```

The previous phase and older full-matrix config files are removed from the
active source tree:

```text
configs/matrix_full.yaml
configs/matrix_phase0.yaml
configs/matrix_phase0_mohr_coulomb.yaml
configs/matrix_phase1.yaml
```

## Active Scripts

The active processing chain is:

```text
01_generate_matrix.py
09_generate_pile_geometry_input_variants.py
02_generate_inputs_from_reference.py
03_submit_slurm_array.sh
04_postprocess_slurm_array.sh
04_postprocess_mcm_array.sh
06_plot_pre_ml_raw_curves.py
07_build_ml_dataset.py
08_train_correction_model.py
```

`02_train_correction_model.py` was the older training entry point and is no
longer part of the active workflow. The current training entry point is
`08_train_correction_model.py`.

## Generated Or Local-Only Folders

These folders contain generated data, cluster outputs, trained models, plots,
or local comparison workspaces and should not be committed:

```text
mgs_rigid_correction/runs/
mgs_rigid_correction/data/extracted/
mgs_rigid_correction/data/processed/
mgs_rigid_correction/plots/
mgs_rigid_correction/results/
Compare_mesh_time_rigid/
Ursprung-Simulation/
```

The two top-level folders `Compare_mesh_time_rigid/` and
`Ursprung-Simulation/` are local comparison/reference workspaces outside the
active ML residual-correction workflow. They are now ignored by `.gitignore`.

## Retained But Not Active

`abaqus/full_geometry_generator/` is retained as an older Abaqus/CAE model
builder. It is not the active input-generation path. The active workflow patches
the checked reference input and the generated non-reference diameter templates.

`docs/ml_residual_correction_plan.md` is retained as background planning
history. It may mention older phase names, `S=1`, and larger geometry matrices.
Use `README.md` and `docs/current_ml_pipeline.md` for the current workflow.

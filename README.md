# MGS Residual Correction

This repository contains the simplified deformable-pile Mass-Gravity-Scaling
residual-correction workflow.

The active project is:

```text
mgs_rigid_correction/
```

The folder name is historical. In the active simplified workflow, the pile is a
deformable steel volume-element pile patched from the checked Abaqus reference
input, not a rigid pile.

Start with:

```text
mgs_rigid_correction/README.md
```

The current ML dataset, training, metrics, and plot outputs are documented in:

```text
mgs_rigid_correction/docs/current_ml_pipeline.md
```

The active validation metrics and publication-style performance plots are based
on grouped out-of-fold predictions by complete density-velocity scenario. The
saved final model is the deployable artifact; final-model curve plots are
diagnostics, not validation evidence.

The full-version input-generation path patches the checked reference Abaqus
`.inp` file. For the CEL setup, the soil mesh/contact definitions remain
unchanged while pile and press coordinates are updated for each geometry. The
Abaqus/CAE model-building generator remains available for later deeper geometry
work:

```text
mgs_rigid_correction/abaqus/full_geometry_generator/
```

Generated Abaqus runs, extracted curves, processed datasets, trained models,
plots, reports, logs, local environments, and local PDFs are ignored by Git.

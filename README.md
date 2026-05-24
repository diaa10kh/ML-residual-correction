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

The active input-generation path patches a checked reference Abaqus `.inp`
file. The previous Abaqus/CAE model-building generator is preserved under:

```text
mgs_rigid_correction/on_hold/complex_abaqus_generator/
```

Generated Abaqus runs, extracted curves, processed datasets, trained models,
plots, reports, logs, local environments, and local PDFs are ignored by Git.

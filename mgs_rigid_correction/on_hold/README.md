# On-Hold Material

This folder contains material that is intentionally not part of the simplified
Phase 0 reference-input workflow.

## moved full geometry generator

The Abaqus/CAE model-building input generator has moved into the active Abaqus
folder:

```text
../abaqus/full_geometry_generator/
```

The simplified Phase 0 workflow still creates Abaqus input files by patching
the checked reference input file with:

```powershell
python scripts\02_generate_inputs_from_reference.py
```

Use the moved generator only after the Abaqus Python model has been checked and
edited for the intended deformable-pile setup.

## training_legacy.yaml

This is the configuration stub for the older non-student ML script sequence.
The active ML sequence is now:

```powershell
python ..\scripts\07_build_ml_dataset.py
python ..\scripts\08_train_correction_model.py
```

Use the generated grouped out-of-fold outputs from the active trainer for
validation metrics and paper figures.

# On-Hold Material

This folder contains material that is intentionally not part of the active
simplified workflow.

## complex_abaqus_generator

The previous Abaqus/CAE model-building input generator is preserved here for
later work. The active workflow now creates Abaqus input files by patching the
checked reference input file with:

```powershell
python scripts\02_generate_inputs_from_reference.py
```

Use the complex generator only when the project returns to full model
construction from Abaqus Python scripts.

## training_legacy.yaml

This is the configuration stub for the older non-student ML script sequence.
The active ML sequence is now:

```powershell
python ..\scripts\07_build_ml_dataset.py
python ..\scripts\08_train_correction_model.py
```

Use the generated grouped out-of-fold outputs from the active trainer for
validation metrics and paper figures.

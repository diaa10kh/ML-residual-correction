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
The active training script is `../scripts/08_train_correction_model.py`.

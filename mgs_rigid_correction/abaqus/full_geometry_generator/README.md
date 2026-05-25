# Full Geometry Abaqus Generator

This folder contains the Abaqus/CAE Python model-building generator used by the
`full-version` workflow. Use it when pile diameter, pile length, or penetration
depth must change.

Prepare case configs and launchers without writing `.inp` files:

```powershell
python abaqus\full_geometry_generator\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata_full.csv --run-id G0_DENS_REF_V_REF_S001 --overwrite-config
```

Generate one representative `.inp` file per geometry and save one `.cae` file
for boundary-condition inspection:

```powershell
python abaqus\full_geometry_generator\02_generate_abaqus_inputs.py --metadata data\extracted\run_metadata_full.csv --one-per-geometry --execute --overwrite-config --save-cae first
```

The representative selection defaults to `DENS_REF`, `V_REF`, and `S=1`, so
this writes one input for each `G0` to `G8`.

Only add `--execute` after the Abaqus model has been checked for the intended
deformable-pile setup.

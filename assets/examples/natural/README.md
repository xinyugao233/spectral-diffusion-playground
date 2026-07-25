# Natural Calibration Images

This directory contains the six frozen, provenance-recorded Experiment 5
source images. Do not add, remove, or replace an image without creating a new
dataset version and updating `../metadata.csv`.

The frozen preprocessing and calibration protocol are documented in
`../../../docs/experiment_05_natural_image_calibration.md`.

Validate the dataset contract before running calibration:

```bash
python scripts/validate_natural_image_dataset.py
```

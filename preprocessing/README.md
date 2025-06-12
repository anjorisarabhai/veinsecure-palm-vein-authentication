# 🧼 Preprocessing Scripts

This folder contains preprocessing functions for the Palm Vein Authentication project.

## Script: `palm_vein_preprocess.py`

- Loads raw palm vein images from `data/raw/`
- Applies:
  - Grayscale conversion
  - Resizing
  - Histogram equalization (CLAHE)
  - Normalization
- Saves processed images to `data/processed/` preserving class-wise folder structure

## Usage

Run the script via terminal:

```bash
python preprocessing/palm_vein_preprocess.py

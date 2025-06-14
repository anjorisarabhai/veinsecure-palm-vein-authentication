# 🧼 Preprocessing Scripts

This folder contains preprocessing scripts and image visualization tools for the Palm Vein Authentication project.

---

## 📄 `palm_vein_preprocess.py`

- Loads raw palm vein images from `data/raw/`
- Applies:
  - Grayscale conversion  
  - Resizing  
  - Histogram equalization (CLAHE)  
  - Normalization  
- Saves processed images to `data/processed/` while preserving class-wise folder structure

---

## 📄 `image_loader_visualizer.py`

- Loads and previews raw palm vein images
- Applies preprocessing steps on a limited sample
- Visualizes:
  - Original vs Processed images (side-by-side)
  - Histogram comparison of pixel intensities

---

## 🔧 Usage

Run the preprocessing pipeline:

```bash
python preprocessing/palm_vein_preprocess.py
```
Run the visualizer (optional sample inspection):
```bash
python preprocessing/image_loader_visualizer.py

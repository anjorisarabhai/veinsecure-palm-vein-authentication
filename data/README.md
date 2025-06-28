# Dataset Information

This folder stores all dataset-related files used in the Palm Vein Authentication project.

⚠️ **Note:** Actual image files (raw or processed) are not included in this repository due to size limits and licensing restrictions.

---

## 📦 Dataset Source

- **Name:** Birjand University Mobile Palmprint Database (BMPD)  
- **Link:** [BMPD Dataset on Kaggle](https://www.kaggle.com/datasets/mahdieizadpanah/birjand-university-mobile-palmprint-databasebmpd)

You must manually download the dataset from Kaggle and extract it into the appropriate folder (see below).

---

## 📁 Folder Structure

data/

├── raw/ # Original dataset as downloaded and extracted

└── processed/ # Cleaned and preprocessed images (e.g., resized, normalized)


- `raw/`: Place the unzipped Kaggle dataset here before running any scripts.
- `processed/`: Generated during preprocessing. This folder will store the output of resize, grayscale, normalization, etc.

---

## Usage Instructions

1. Download the dataset manually from the Kaggle link above.
2. Place the unzipped content into `data/raw/`.
3. Run the preprocessing pipeline:

```bash
python preprocessing/palm_vein_preprocess.py
```
This will save preprocessed images to `data/processed/`.

---

## ❗ Do Not Upload Large Files

To keep the repository lightweight do **not** push raw or processed image files to GitHub.

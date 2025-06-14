# VeinSecure Palm Vein Authentication

This project aims to build a deep learning-based palm vein biometric authentication system.

## Project Overview

- Use palm vein images to identify individuals.
- Apply image preprocessing techniques like CLAHE and resizing.
- Train convolutional neural networks for recognition.
- Evaluate model performance with metrics like accuracy and ROC curves.

## Dataset

- Using the Birjand University Mobile Palmprint Database (BMPD) from Kaggle.
- Dataset is stored in the `data/raw/` folder.

## Structure

- `data/` - Dataset files
- `preprocessing/` - Scripts to preprocess images
- `model/` - Model training and evaluation scripts
- `notebooks/` - Jupyter notebooks for EDA and experiments
- `utils/` - Utility functions
- `frontend/` - Frontend application code (if applicable)
- `api/` - API backend code (if applicable)

## How to Run

1. Install dependencies:  
   `pip install -r requirements.txt`

2. Run preprocessing:  
   `python preprocessing/preprocess.py`

3. Explore notebooks for analysis.

---

## Contributors

- Anjori Sarabhai  
- Apoorva Kashyap


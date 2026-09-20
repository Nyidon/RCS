# Throat + Belly Biometric Extraction & Standardisation Pipeline

This directory contains the standalone **Phase 1 (YOLOv8 + SAM 2 Extraction)** and **Phase 2 (Geometric & Photometric Preprocessing)** for combined **Throat + Belly (full ventral)** toad pattern re-identification.

---

## 📁 Directory Structure

```
throat_belly/
├── data/
│   ├── raw/                        # 581 raw survey photos
│   ├── train/ & val/               # Partitioned sets + LabelMe JSON annotations (.json + .txt)
│   ├── SAM2_Data/                  # Full ventral crops extracted by SAM 2
│   └── preprocessed/               # Standardised 256x256 RGB ventral crops
│
├── src/
│   ├── dataset.yaml                # Ultralytics YOLO configuration
│   ├── yolo_train.py               # Trains YOLOv8n and prompts SAM 2 with ventral masks
│   └── preprocessing.py            # Symmetry sweep, Chin-North orientation, CLAHE & USM
│
└── utils/
    ├── split_data.py               # Splits raw images into train/ and val/
    └── convert_to_txt.py           # Converts LabelMe JSON annotations to YOLO format
```

---

## 🚀 Step-by-Step Execution

### 1. Data Preparation
```bash
# Split raw images into train and val
python 01-02_Throat_Belly_pipeline/utils/split_data.py

# Convert LabelMe JSON annotations into YOLO TXT format
python 01-02_Throat_Belly_pipeline/utils/convert_to_txt.py
```

### 2. YOLOv8 Training & SAM 2 Extraction (Phase 1)
```bash
# Train YOLO detector and extract smooth ventral crops via SAM 2
python 01-02_Throat_Belly_pipeline/src/yolo_train.py
```

### 3. Pose Standardisation & CLAHE Enhancement (Phase 2)
```bash
# Generate standardized 256x256 RGB crops with strict black backgrounds
python 01-02_Throat_Belly_pipeline/src/preprocessing.py
```

Outputs from `data/preprocessed/` can then be directly linked or copied to `03_2_Siamese_network/data/train/rgb_throat_belly_id/` for Phase 3 deep metric learning.

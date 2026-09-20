# RCS: Automated Toad Biometric Re-Identification Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8%20%2F%20SAM2-Computer%20Vision-green.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An automated computer vision and deep biometric metric learning framework for individual amphibian re-identification (*Bufo bufo* / *Bombina variegata*) in ecological Capture-Mark-Recapture (CMR / RCS) surveys. The system identifies individual toads across temporal field survey sessions using the natural, unique pigmentation spot patterns located on their subgular throat and ventral skin regions.

---

## 📌 Architecture & Data Flow

```
Raw Survey Photos (Aulbachtal / Hochfläche)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  Phase 1: Detection & Segmentation (YOLOv8 + SAM 2)    │
│  • YOLOv8n detector predicts anatomical bounding boxes │
│  • SAM 2 extracts smooth triangular masks              │
│  • Chaikin curve smoothing & 2.5% border inset        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Standardisation & Enhancement                │
│  • Multi-Scale Bilateral Symmetry Sweep (0° to 180°)   │
│  • 4-Feature Anatomical Orientation Scorer (Strict N)  │
│  • LAB CLAHE Lightness enhancement & Unsharp Masking   │
└──────────────┬───────────────────────────┬─────────────┘
               │                           │
               ▼                           ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ Phase 3.1: Bi-Model Consensus│ │ Phase 3.2: Deep Metric Net   │
│ • SIFT (L2) + AKAZE (Binary) │ │ • ConvNeXt-Tiny Backbone     │
│ • Boundary Mask Erosion      │ │ • Batch-Hard Triplet Mining  │
│ • Lowe's Ratio Test (τ=0.80) │ │ • Cosine Margin Loss (m=0.4) │
│ • 2D Affine RANSAC Consensus │ │ • 256-d Metric Embeddings    │
│ • Self-Similarity Normalise  │ │ • 0° / 180° Test-Time Aug    │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────┐
│         Automated Evaluation & Benchmarks              │
│  1. Recapture Identification (100 Images, 40 IDs)      │
│  2. False Merge Prevention (60 Unique Individuals)     │
│  3. Multi-Model Comparison (Siamese vs. Bi-Model vs.   │
│     WildID Baseline)                                   │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Components

### Phase 1: Throat Detection & Fine Segmentation (`01_Throat_detection_pipeline/`)
* **YOLOv8n Detector**: Trained on localized subgular skin regions to predict tightly bounded neck boxes.
* **SAM 2 Prompting (`sam2_t.pt`)**: Uses bounding box prompts to generate high-solidity semantic masks with adaptive perimeter erosion ($2.5\%$) and Chaikin vertex smoothing, achieving **$97.17\%$ border purity**.

### Phase 2: Geometric & Photometric Standardisation (`02_Throat_Preprocessing/`)
* **Bilateral Symmetry Sweep**: Sweeps angles from $0^\circ$ to $180^\circ$ (coarse $2^\circ$, fine $0.5^\circ$) maximizing vertical Jaccard overlap between mirrored left/right mask halves.
* **4-Feature Anatomical Orientation Scorer**: Eliminates $180^\circ$ upside-down ambiguity using longitudinal width gradient, corner voids, area mass distribution, and center-of-mass moments.
* **Photometric Enhancement**: LAB color-space bilateral filtering, CLAHE ($L$-channel), and unsharp detail enhancement to highlight subtle carotenoid pigments and melanin spots against specular glare.

### Phase 3.1: Bi-Model Deterministic Consensus (`03_1_Bi_model_consensus/`)
* **Complementary Descriptors**: Fuses **SIFT** (128-d gradient histograms) and **AKAZE** (binary MLDB) with boundary mask erosion (13 px).
* **Spatial Verification**: 2-NN Lowe's Ratio Test ($\tau = 0.80$) and 2D Affine RANSAC geometric consensus filtering.
* **Self-Similarity Normalization ($R_{ij} = S_{ij} / S_{ii}$)**: Calibrates matching scores onto a $[0, 1]$ scale invariant to spot density.

### Phase 3.2: Deep Metric Learning (`03_2_Siamese_network/`)
* **Backbone Architecture**: `convnext_tiny` pre-trained on wildlife representations with an $L_2$-normalized 256-d projection head.
* **Loss & Optimization**: Batch-Hard Triplet Mining ($P=8, K=4$) with Cosine Triplet Margin Loss ($\text{margin} = 0.4$) and $0^\circ/180^\circ$ Test-Time Augmentation (TTA).

---

## 📊 Empirical Benchmarks

### 1. Recapture Identification Benchmark (100 Images, 40 Identities, 60 True Recaptures)

| Model Architecture | Target Anatomy | Top-1 Recapture | Top-3 Recapture | Top-5 Recapture | Top-10 Recapture | Top-20 Recapture | Mean Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese ConvNeXt-Tiny (RGB)** | **Throat Only** | **71.67% (43/60)** | **90.00% (54/60)** | **91.67% (55/60)** | **96.67% (58/60)** | **98.33% (59/60)** | **1.97** |
| **WildID Baseline** | Throat Only | **71.67% (43/60)** | 80.00% (48/60) | 83.33% (50/60) | 91.67% (55/60) | 96.67% (58/60) | 2.53 |
| **Siamese ConvNeXt-Tiny (Gray)**| Throat Only | 68.33% (41/60) | 75.00% (45/60) | 86.67% (52/60) | 91.67% (55/60) | 96.67% (58/60) | 2.50 |
| **Bi-Model Consensus (SIFT+AKAZE)** | **Throat + Belly** | **96.67% (58/60)** | **96.67% (58/60)** | **98.33% (59/60)** | **100.00% (60/60)**| **100.00% (60/60)**| **1.15** |
| **Bi-Model Consensus (SIFT+AKAZE)** | Throat Only | 41.67% (25/60) | 51.67% (31/60) | 61.67% (37/60) | 71.67% (43/60) | 85.00% (51/60) | 4.78 |

---

### 2. Strict Unique Rejection Benchmark (60 Unique Individuals — False Merge Prevention)

| Model Architecture | Target Anatomy | Operating Threshold | Specificity (TNR %) | False Merge Rate (FAR %) | Max Impostor ($S_{\max}$) | Safe Zero-FP Cutoff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bi-Model Consensus** | **Throat Only** | $T^* = 0.060$ | **100.00% (60/60)** | **0.00% (0/60)** | **0.0578** | $T^*_{\text{safe}} \ge 0.058$ |
| **Bi-Model Consensus** | **Throat + Belly** | $T^* = 0.060$ | **96.67% (58/60)** | **3.33% (2/60)** | **0.1341** | $T^*_{\text{safe}} \ge 0.135$ |
| **Siamese ConvNeXt-Tiny (RGB)** | Throat Only | $\tau = 0.700$ | 80.00% (48/60) | 20.00% (12/60) | 0.8792 | $\tau_{\text{safe}} \ge 0.880$ |
| **Siamese ConvNeXt-Tiny (Gray)**| Throat Only | $\tau = 0.700$ | 73.33% (44/60) | 26.67% (16/60) | 0.8180 | $\tau_{\text{safe}} \ge 0.818$ |

---

## 📁 Repository Structure

```
rcs_pipeline/
├── 01_Throat_detection_pipeline/                   # Phase 1: Object detection & mask extraction
│   ├── src/
│   │   ├── dataset.yaml                            # YOLO dataset configuration
│   │   ├── yolo_train.py                           # YOLOv8n training & SAM 2 crop generation
│   │   └── evaluate_sam2.py                        # Segmentation overlap & border quality benchmark
│   └── utils/
│       ├── convert_to_txt.py                       # LabelMe JSON -> YOLO TXT annotation converter
│       └── split_data.py                           # Survey dataset flattener & train/val splitter
│
├── 02_Throat_Preprocessing/                        # Phase 2: Pose standardisation & enhancement
│   └── src/
│       ├── preprocessing.py                        # Grayscale bilateral symmetry & CLAHE
│       └── preprocessing_rgb.py                    # RGB LAB color-space CLAHE & unsharp boost
│
├── 03_1_Bi_model_consensus/                        # Phase 3.1: Bi-Model Consensus Engine
│   ├── README.md                                   # Phase 3.1 documentation
│   └── src/
│       ├── feature_extractor.py                    # Dual SIFT + AKAZE extractor & spatial RANSAC
│       ├── pairwise_matcher.py                     # Multi-threaded NxN similarity calculator
│       └── consensus_engine.py                     # Bi-model fusion & Top-K candidate ranking
│
├── 03_2_Siamese_network/                           # Phase 3.2: Deep Metric Learning
│   ├── README.md                                   # Phase 3.2 documentation
│   └── src/
│       ├── dataset.py                              # Triplet mining & identity cluster loaders
│       ├── model.py                                # ConvNeXt-Tiny with L2 projection head
│       ├── train.py                                # Batch-Hard metric learning CLI
│       ├── infer_top20.py                          # Top-20 candidate retrieval engine
│       ├── run_test_inference.py                   # Recapture evaluation benchmark
│       ├── compare_all_models.py                   # 4-Way comparative benchmarking CLI
│       └── test_strict_unique_rejection.py         # False merge evaluation on 60 unique toads
│
├── results/                                        # Evaluation metrics, predictions & plots
│   ├── evaluation/                                 # Multi-sheet Excel & JSON benchmark reports
│   └── plots/                                      # Publication-ready figures & comparison charts
│
├── .gitignore                                      # Excludes large binaries, checkpoints, & raw data
├── main.py                                         # Workspace entry point
└── README.md                                       # Main repository documentation
```

---

## 🚀 Step-by-Step Execution Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Nyidon/RCS.git
cd RCS

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision timm ultralytics opencv-python numpy pandas matplotlib tqdm openpyxl faiss-cpu scikit-learn
```

### 2. Preprocessing & Alignment (Grayscale & RGB)
```bash
# Run Grayscale Preprocessing
python 02_Throat_Preprocessing/src/preprocessing.py

# Run RGB LAB Preprocessing
python 02_Throat_Preprocessing/src/preprocessing_rgb.py
```

### 3. Deep Metric Learning Training
```bash
# Train RGB Siamese Network (ConvNeXt-Tiny)
python 03_2_Siamese_network/src/train.py --mode rgb --epochs 25

# Train Grayscale Siamese Network
python 03_2_Siamese_network/src/train.py --mode gray --epochs 25
```

### 4. Running Benchmarks
```bash
# Run 100-Image Recapture Benchmark (Siamese RGB & Bi-Model Consensus)
python 03_2_Siamese_network/src/run_test_inference.py --model both --mode rgb

# Comprehensive Comparison (Siamese vs. Bi-Model vs. WildID Baseline)
python 03_2_Siamese_network/src/compare_all_models.py

# Evaluate False Merge Resistance on 60 Unique Individuals
python 03_2_Siamese_network/src/test_strict_unique_rejection.py --model both
```

---

## ⚙️ Hardware Acceleration
Optimized for Apple Silicon hardware acceleration via PyTorch `mps` (`device='mps'`), automatically falling back to CUDA GPUs or multi-core CPU threads.
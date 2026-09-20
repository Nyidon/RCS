# RCS: Automated Toad Biometric Re-Identification Pipeline

An automated computer vision and deep biometric metric learning framework for individual amphibian re-identification (*Bufo bufo* / *Bombina variegata*) in ecological Capture-Mark-Recapture (CMR / RCS) surveys. The system identifies individual toads across temporal field survey sessions using the natural, unique pigmentation spot patterns located on their subgular throat and full ventral skin regions.

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
│  Phase 2: Geometric & Photometric Standardisation      │
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
│  SAM2 performance                                      │
│  RBG Vs Grayscaled data                                │
│  On Throat alone across different models:              │
│    1. Recapture Identification (100 Images, 40 IDs)    │
│    2. False Merge Prevention (60 Unique Individuals)   │
│  Anatomical Comparison of Throat vs. Throat + Belly:   │
│     Bi-Model Consensus Vs Deep Metric Net              │
│    1. Recapture Identification (100 Images, 40 IDs)    │
│    2. False Merge Prevention (60 Unique Individuals)   │
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

> **Note on Throat + Belly Pipeline (`01-02_Throat_Belly_pipeline/`)**:  
> The `01-02_Throat_Belly_pipeline/` directory is an end-to-end unified stage executing YOLOv8 detection, SAM 2 segmentation, and bilateral standardization directly on the full ventral region (Throat + Belly combined) rather than the throat alone. This is done so to compare whether throat results better output than throat + belly.

### Extension: Anatomical comparison on Throat and Throat + Belly  
* **Bi-Model Consensus and Deep Metric Learning performances on throat and throat + belly.

---

## 📊 Empirical Benchmarks

### 1. SAM 2 Semantic Mask Segmentation Benchmark ($N=70$ Polygons)

The automated throat mask extraction pipeline was evaluated against $N=70$ human ground-truth LabelMe polygonal annotations from diverse field survey sessions.

| Metric | Measured Value | Standard Deviation | Description |
| :--- | :---: | :---: | :--- |
| **YOLOv8n Throat Detection Rate** | **100.00% (70/70)** | — | Perfect box localization across lighting & poses |
| **Mean Mask IoU (Jaccard Index)** | **73.14%** | $\pm 6.71\%$ | Spatial overlap with human anatomical polygon |
| **Median Mask IoU** | **74.06%** | — | Robust median spatial overlap |
| **Mean Dice Coefficient ($F_1$)** | **84.30%** | $\pm 4.71\%$ | Harmonic mean of precision and recall |
| **Mean Border Precision** | **97.17%** | $\pm 2.45\%$ | Strict exclusion of substrate/background pixels |
| **Mean Coverage / Recall** | **75.07%** | $\pm 8.12\%$ | Percentage of manual ground-truth patch captured |
| **Mean SAM 2 Mask Solidity** | **0.9883** | $\pm 0.009$ | High shape compactness with smooth Chaikin borders |

---

### 2. Photometric Representation Ablation: RGB (LAB CLAHE) vs. Grayscale

| Metric | Siamese ConvNeXt (RGB) | Siamese ConvNeXt (Grayscale) | Performance Delta ($\Delta_{\text{RGB}}$) |
| :--- | :---: | :---: | :---: |
| **Top-1 Recapture Accuracy** | **71.67% (43/60)** | 68.33% (41/60) | **+3.34%** |
| **Top-3 Recapture Accuracy** | **90.00% (54/60)** | 75.00% (45/60) | **+15.00%** |
| **Top-5 Recapture Accuracy** | **91.67% (55/60)** | 86.67% (52/60) | **+5.00%** |
| **Top-10 Recapture Accuracy** | **96.67% (58/60)** | 91.67% (55/60) | **+5.00%** |
| **Top-20 Recapture Accuracy** | **98.33% (59/60)** | 96.67% (58/60) | **+1.66%** |
| **Mean Recapture Rank** | **1.97** | 2.50 | **-0.53** *(Better)* |
| **Mean Average Precision (mAP)** | **78.37%** | 74.85% | **+3.52%** |
| **Unique Toad Rejection Specificity (TNR)** | **80.00% (48/60)** | 73.33% (44/60) | **+6.67%** |
| **False Merge Rate (FAR)** | **20.00% (12/60)** | 26.67% (16/60) | **-6.67%** *(Lower)* |

---

### 3. Throat-Only Recapture Identification Benchmark (100 Images, 40 Identities, 60 True Recaptures)

| Model Architecture | Target Anatomy | Top-1 Recapture | Top-3 Recapture | Top-5 Recapture | Top-10 Recapture | Top-20 Recapture | Mean Rank | mAP (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Siamese ConvNeXt-Tiny (RGB)** | **Throat Only** | **71.67% (43/60)** | **90.00% (54/60)** | **91.67% (55/60)** | **96.67% (58/60)** | **98.33% (59/60)** | **1.97** | **78.37%** |
| **WildID Baseline** | Throat Only | **71.67% (43/60)** | 80.00% (48/60) | 83.33% (50/60) | 91.67% (55/60) | 96.67% (58/60) | 2.53 | 71.67% |
| **Siamese ConvNeXt-Tiny (Gray)**| Throat Only | 68.33% (41/60) | 75.00% (45/60) | 86.67% (52/60) | 91.67% (55/60) | 96.67% (58/60) | 2.50 | 74.85% |
| **Bi-Model Consensus (SIFT+AKAZE)** | Throat Only | 41.67% (25/60) | 51.67% (31/60) | 61.67% (37/60) | 71.67% (43/60) | 85.00% (51/60) | 4.78 | 45.93% |

---

### 4. Strict Unique Rejection Benchmark (60 Unique Individuals — False Merge Prevention)

| Model Architecture | Target Anatomy | Operating Threshold | Specificity (TNR %) | False Merge Rate (FAR %) | Max Impostor ($S_{\max}$) | Safe Zero-FP Cutoff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bi-Model Consensus** | **Throat Only** | $T^* = 0.060$ | **100.00% (60/60)** | **0.00% (0/60)** | **0.0578** | $T^*_{\text{safe}} \ge 0.058$ |
| **Bi-Model Consensus** | **Throat + Belly** | $T^* = 0.060$ | **96.67% (58/60)** | **3.33% (2/60)** | **0.1341** | $T^*_{\text{safe}} \ge 0.135$ |
| **Siamese ConvNeXt-Tiny (RGB)** | Throat Only | $\tau = 0.700$ | 80.00% (48/60) | 20.00% (12/60) | 0.8792 | $\tau_{\text{safe}} \ge 0.880$ |
| **Siamese ConvNeXt-Tiny (Gray)**| Throat Only | $\tau = 0.700$ | 73.33% (44/60) | 26.67% (16/60) | 0.8174 | $\tau_{\text{safe}} \ge 0.818$ |
| **Siamese ConvNeXt-Tiny (RGB)** | Throat + Belly | $\tau = 0.700$ | 66.67% (40/60) | 33.33% (20/60) | 0.9287 | $\tau_{\text{safe}} \ge 0.930$ |

---

### 5. Anatomical Factorial Comparison: Throat Alone vs. Throat + Belly ($2 \times 2$ Analysis)

A central ecological question is whether survey photographs should isolate the subgular throat or capture the entire ventral surface (throat + belly combined).

| Model Family | Anatomical Target | Recapture Top-1 | Recapture Top-5 | Recapture Top-10 | Recapture Top-20 | Mean Rank | Rejection Specificity (TNR) | False Merge Rate (FAR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bi-Model Consensus** | **Throat Only** | 41.67% | 61.67% | 71.67% | 85.00% | 4.78 | **100.00%** | **0.00%** |
| **Bi-Model Consensus** | **Throat + Belly** | **96.67%** | **98.33%** | **100.00%** | **100.00%** | **1.15** | **96.67%** | **3.33%** |
| *Consensus Delta* | — | **+55.00%** | **+36.66%** | **+28.33%** | **+15.00%** | **-3.63** | *-3.33%* | *+3.33%* |
| **Siamese ConvNeXt (RGB)** | **Throat Only** | **71.67%** | **91.67%** | **96.67%** | **98.33%** | **1.97** | **80.00%** | **20.00%** |
| **Siamese ConvNeXt (RGB)** | **Throat + Belly** | 61.67% | 88.33% | **96.67%** | **98.33%** | 2.49 | 66.67% | 33.33% |
| *Siamese Delta* | — | **-10.00%** | *-3.34%* | *0.00%* | *0.00%* | *+0.52* | *-13.33%* | *+13.33%* |

#### 🔬 Why the Models Diverge Across Anatomical Targets:
1. **Bi-Model Consensus leaps $+55.00\%$ on Throat + Belly ($96.67\%$ Top-1)**:  
   Local feature matchers (SIFT/AKAZE) thrive on high spot counts. Expanding the patch across the belly provides abundant anchor points ($60\text{--}120$ keypoints vs. $15\text{--}25$ on throat only), enabling RANSAC to consistently establish geometric consensus even under slight camera tilt.
2. **Siamese Network excels on Throat Only ($71.67\%$ Top-1 vs. $61.67\%$)**:  
   Deep metric learning models resize inputs into fixed square dimensions ($256 \times 256$). Squishing elongated $1:2.5$ ventral rectangles blurs subtle micro-dots and introduces background aspect distortions, whereas compact triangular throat crops preserve pristine pixel resolution.

---

## 📁 Repository Structure

```
rcs_pipeline/
├── 01_Throat_detection_pipeline/                   # Phase 1: Throat BBox detection & SAM 2 extraction
│   ├── src/
│   │   ├── dataset.yaml                            # YOLO dataset configuration
│   │   ├── yolo_train.py                           # YOLOv8n training & SAM 2 crop generation
│   │   └── evaluate_sam2.py                        # Segmentation overlap & border quality benchmark
│   └── utils/
│       ├── convert_to_txt.py                       # LabelMe JSON -> YOLO TXT annotation converter
│       └── split_data.py                           # Survey dataset flattener & train/val splitter
│
├── 01-02_Throat_Belly_pipeline/                    # Unified YOLO + SAM 2 + Preprocessing for Throat+Belly
│   ├── src/
│   │   ├── yolo_train.py                           # Ventral bounding box detector & SAM 2 segmenter
│   │   └── preprocessing.py                        # Full ventral bilateral alignment & enhancement
│   └── utils/
│       ├── convert_to_txt.py                       # Annotation converter for ventral patches
│       └── split_data.py                           # Ventral train/val splitter
│
├── 02_Throat_Preprocessing/                        # Phase 2: Throat pose standardisation & enhancement
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

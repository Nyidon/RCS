# Stage 3.2: Siamese Deep Metric Learning & Automated Benchmarks

## 📌 Overview

**Stage 3.2 (`03_2_Siamese_network`)** fine-tunes deep neural embeddings on verified toad identity clusters using **Triplet Margin Metric Learning with Cosine Distance**.

Rather than relying on generic vision features or 1:1 pairwise thresholds, the fine-tuned Siamese network maps individual toad ventral patterns onto an identity hypersphere $\mathbb{S}^{d-1}$ where:
- Ventral spot patterns from the **same toad** cluster tightly with high cosine similarity.
- Ventral patterns from **different toads** are pushed apart beyond a learned separation margin $\alpha$.

---

## 🏗️ Architecture & Component Map

```
03_2_Siamese_network/
├── checkpoints/                    # Saved model weights (best_siamese_model.pt)
│   ├── rgb/
│   └── gray/
├── data/                           # Curated dataset splits & pools
│   ├── train/                      # 250 curated toad identity clusters (422 images)
│   │   ├── rgb_toad_id/            # Multi-sighting (M001..M100) + singletons (S001..S150)
│   │   ├── gray_toad_id/
│   │   ├── rgb_toad_id_manifest.json
│   │   └── gray_toad_id_manifest.json
│   ├── test/                       # 100 test images with 40 clusters (evaluation benchmark)
│   │   ├── rgb_test/               # Flat RGB test images
│   │   ├── gray_test/              # Flat Grayscale test images
│   │   ├── test_manifest.json      # Ground truth test manifest
│   │   └── wildID/                 # Ground truth WildID baseline files
│   ├── unique_toads/               # 60 human-verified unique toads (false merge benchmark)
│   │   ├── rgb_strict_unique/      # Flat RGB unique toad images
│   │   └── gray_strict_unique/     # Flat Grayscale unique toad images
│   ├── rgb/                        # Full pool of 1,193 preprocessed RGB crops
│   └── gray/                       # Full pool of 1,193 preprocessed Grayscale crops
├── src/
│   ├── dataset.py                  # Identity cluster loader & Triplet generator (A, P, N)
│   ├── model.py                    # Deep metric network with L2 projection head & Cosine Triplet Loss
│   ├── train.py                    # Metric learning training CLI (MPS/CUDA/CPU)
│   ├── infer_top20.py              # Top-20 similarity ranking engine across gallery
│   ├── run_test_inference.py       # Benchmark evaluation on 100 test images (40 clusters)
│   ├── compare_all_models.py       # Comparative evaluation: Siamese vs Bi-Model vs WildID
│   └── test_strict_unique_rejection.py # False merge evaluation on 60 unique toads
└── README.md                       # Stage 3.2 technical guide
```

---

## 🚀 Step-by-Step Execution

### 1. Train the Siamese Metric Network
Train the network on the confirmed identity clusters:

```bash
# Train on RGB track (default 25 epochs)
python 03_2_Siamese_network/src/train.py --mode rgb --epochs 25

# Train on Grayscale track
python 03_2_Siamese_network/src/train.py --mode gray --epochs 25
```

---

### 2. Run Test Inference & Candidate Ranking
Generate Top-20 ranked candidates for each sequential sighting in the test dataset:

```bash
# Run test inference on both Siamese (RGB) and Bi-Model Consensus
python 03_2_Siamese_network/src/run_test_inference.py --model both --mode rgb
```

---

### 3. Comprehensive Benchmark Comparison (Siamese vs Bi-Model vs WildID)
Compare all models against ground-truth recapture manifests:

```bash
python 03_2_Siamese_network/src/compare_all_models.py
```

---

### 4. Strict Unique Toad Rejection Benchmark (False Merge Prevention)
Evaluate specificity and false-merge prevention across strictly unique individuals:

```bash
python 03_2_Siamese_network/src/test_strict_unique_rejection.py --model both --mode rgb
```

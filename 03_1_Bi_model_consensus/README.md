# Stage 3.1: Bi-Model Consensus Engine & Dynamic Toad Catalog

## 📌 Overview
`03_1_Bi_model_consensus` provides high-precision candidate retrieval, dual feature extraction (SIFT + AKAZE), ratio-normalized bi-model consensus scoring, and an interactive Streamlit human verification interface for individual amphibian re-identification.

---

## 🎯 Key Capabilities

1. **Dual Feature Extraction & Pairwise Ratio Matching**:
   - Primary: SIFT (128-d L2 Gradient, Lowe's Ratio $\tau=0.80$, RANSAC Spatial Consensus)
   - Secondary: AKAZE / Binary (MLDB, Hamming Distance)
   - Self-Similarity Normalization: $R_{ij} = S_{ij} / S_{ii}$
2. **Bi-Model Consensus Scoring**:
   $$R_{ij}^{\text{consensus}} = 0.55 \cdot R_{ij}^{\text{SIFT}} + 0.45 \cdot R_{ij}^{\text{Binary}}$$
   Symmetrized conservative matrix:
   $$S_{ij}^{\text{sym}} = \min\left(R_{ij}^{\text{consensus}}, R_{ji}^{\text{consensus}}\right)$$
3. **Top-20 Candidate Retrieval Engine**:
   - **Strict Sequential WildID Protocol**: Replicates WildID's incremental field arrival protocol ($\min(i, 20)$ candidates chosen strictly from past encounters $0 \dots i-1$).
4. **Interactive Side-by-Side Review UI (`app_consensus_ui.py`)**:
   - Side-by-side spot visualizer with live correspondence lines.
   - One-click confirmation buttons (`[✅ Confirm Match]` / `[🆕 Register New Individual]`).
   - **Zero Data Loss**: Hidden `.cache/` auto-checkpointing prevents progress loss on browser reload or interruption.
   - **In-Memory Excel Export**: Downloads multi-sheet `.xlsx` workbooks directly to your browser without cluttering local disk.

---

## 🚀 Running the Consensus UI
```bash
streamlit run 03_1_Bi_model_consensus/src/app_consensus_ui.py
```

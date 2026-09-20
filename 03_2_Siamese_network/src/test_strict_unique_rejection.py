#!/usr/bin/env python3

# Strict Unique Toad Rejection & False Merge Prevention Benchmark

import os
import sys
import json
import time
import argparse
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
consensus_src = project_root / "03_1_Bi_model_consensus" / "src"

for path_dir in [str(current_dir), str(consensus_src), str(project_root)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

from infer_top20 import load_siamese_model, extract_siamese_embedding
from dataset import get_default_transforms

try:
    from consensus_engine import BiModelConsensusEngine
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("consensus_engine", str(consensus_src / "consensus_engine.py"))
    if spec and spec.loader:
        consensus_module = importlib.util.module_from_spec(spec)
        sys.modules["consensus_engine"] = consensus_module
        spec.loader.exec_module(consensus_module)
        BiModelConsensusEngine = consensus_module.BiModelConsensusEngine
    else:
        raise ImportError(f"Could not import BiModelConsensusEngine from {consensus_src}")


def load_unique_toad_images(data_dir: Path):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    files = sorted([
        unicodedata.normalize('NFC', f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(valid_exts) and not f.startswith('.')
    ])
    paths = [data_dir / f for f in files]
    return files, paths


def compute_siamese_similarity_matrix(paths, siamese_root: Path, mode: str = "rgb", checkpoint_path: Path = None):
    print(f"⚡ Computing Siamese Network (ConvNeXt-Tiny [{mode.upper()}]) embeddings for unique toads...")
    t0 = time.time()
    if checkpoint_path is None:
        checkpoint_path = siamese_root / "checkpoints" / mode / "best_siamese_model.pt"
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"📦 Loading Siamese Weights from: {checkpoint_path}")
    model = load_siamese_model(checkpoint_path=checkpoint_path, mode=mode, backbone="convnext_tiny")
    transform = get_default_transforms(is_train=False)

    embs_0 = []
    embs_180 = []
    for p in paths:
        e0, e180 = extract_siamese_embedding(p, model=model, transform=transform, return_tta=True)
        embs_0.append(e0)
        embs_180.append(e180)

    mat_0 = torch.stack(embs_0)
    mat_180 = torch.stack(embs_180)

    sim_0 = torch.mm(mat_0, mat_0.t()).numpy()
    sim_180 = torch.mm(mat_180, mat_0.t()).numpy()
    sim_matrix = np.maximum(sim_0, sim_180)
    elapsed = time.time() - t0
    print(f"✅ Siamese matrix computed ({len(paths)}x{len(paths)}) in {elapsed:.2f}s")
    return sim_matrix, elapsed


def compute_consensus_similarity_matrix(paths):

    print("⚡ Computing Bi-Model Consensus (AKAZE + SIFT) pairwise matrices for unique toads...")
    t0 = time.time()
    engine = BiModelConsensusEngine(akaze_weight=0.55, sift_weight=0.45, threshold_t_star=0.080)
    batch_data = engine.precompute_batch_matrices(paths)
    sim_matrix = batch_data["R_consensus"]
    elapsed = time.time() - t0
    print(f"✅ Consensus matrix computed ({len(paths)}x{len(paths)}) in {elapsed:.2f}s")
    return sim_matrix, elapsed


def evaluate_pure_unique_metrics(sim_matrix, files, model_name, default_threshold):
    N = len(files)

    # -------------------------------------------------------------------------
    # 1. Sequential Online Cataloging Simulation
    # -------------------------------------------------------------------------
    seq_decisions = []
    discovered_clusters = 0
    seq_false_merges = 0
    cumulative_clusters = []

    for i in range(N):
        q_file = files[i]

        if i == 0:
            discovered_clusters += 1
            cumulative_clusters.append(discovered_clusters)
            seq_decisions.append({
                "Image_Index": i + 1,
                "Image_File": q_file,
                "Comparison_Gallery_Size": 0,
                "Top_Candidate_Image": "NONE (First Sighting)",
                "Max_Similarity_Score": 0.0,
                "Threshold": default_threshold,
                "Decision": "NEW_TOAD_CREATED",
                "Is_Correct_Rejection": True,
                "Is_False_Merge": False,
                "Assigned_Toad_Cluster": f"Toad_{discovered_clusters:03d}"
            })
            continue

        past_scores = [sim_matrix[i, j] for j in range(i)]
        best_past_idx = int(np.argmax(past_scores))
        max_score = float(past_scores[best_past_idx])
        top_cand_file = files[best_past_idx]

        if max_score < default_threshold:
            discovered_clusters += 1
            decision = "NEW_TOAD_CREATED"
            is_correct = True
            is_false_merge = False
            assigned_cluster = f"Toad_{discovered_clusters:03d}"
        else:
            seq_false_merges += 1
            decision = f"FALSE_MERGE_WITH_{files[best_past_idx]}"
            is_correct = False
            is_false_merge = True
            assigned_cluster = f"Merged_with_Img_{best_past_idx+1}"

        cumulative_clusters.append(discovered_clusters)

        seq_decisions.append({
            "Image_Index": i + 1,
            "Image_File": q_file,
            "Comparison_Gallery_Size": i,
            "Top_Candidate_Image": top_cand_file,
            "Max_Similarity_Score": max_score,
            "Threshold": default_threshold,
            "Decision": decision,
            "Is_Correct_Rejection": is_correct,
            "Is_False_Merge": is_false_merge,
            "Assigned_Toad_Cluster": assigned_cluster
        })

    # Sequential Specificity and FAR
    tnr_seq = ((N - seq_false_merges) / N) * 100.0
    far_seq = (seq_false_merges / N) * 100.0

    # -------------------------------------------------------------------------
    # 2. Complete All-vs-All Impostor Analysis (60x60, 1,770 pairs)
    # -------------------------------------------------------------------------
    matrix_no_diag = sim_matrix.copy()
    np.fill_diagonal(matrix_no_diag, -9999.0)

    max_impostor_scores = np.max(matrix_no_diag, axis=1)
    worst_cand_indices = np.argmax(matrix_no_diag, axis=1)

    all_pairs_log = []
    for i in range(N):
        all_pairs_log.append({
            "Image_Index": i + 1,
            "Image_File": files[i],
            "Worst_Impostor_Candidate": files[worst_cand_indices[i]],
            "Max_Impostor_Score": float(max_impostor_scores[i]),
            "Threshold": default_threshold,
            "Rejected_at_Threshold": bool(max_impostor_scores[i] < default_threshold)
        })

    # All-vs-All TNR and FAR across all 60 toads against the entire set
    tnr_all = float(np.mean(max_impostor_scores < default_threshold) * 100.0)
    far_all = float(np.mean(max_impostor_scores >= default_threshold) * 100.0)

    # Metric 4: Max-Impostor Distribution stats
    mu_imp = float(np.mean(max_impostor_scores))
    median_imp = float(np.median(max_impostor_scores))
    std_imp = float(np.std(max_impostor_scores))
    worst_case_imp = float(np.max(max_impostor_scores))
    min_imp = float(np.min(max_impostor_scores))

    # Metric 5: Safe Zero-False-Positive Threshold (τ_safe)
    safe_zero_fp_threshold = float(np.max(max_impostor_scores)) + 1e-4

    # Threshold Specificity Sweep
    if "Consensus" in model_name:
        threshold_sweep = np.linspace(0.01, 0.12, 100)
    else:
        threshold_sweep = np.linspace(0.30, 0.95, 100)

    specificity_curve = []
    for t_val in threshold_sweep:
        tnr_val = (np.sum(max_impostor_scores < t_val) / N) * 100.0
        specificity_curve.append(tnr_val)

    return {
        "model_name": model_name,
        "total_unique_toads": N,
        "operating_threshold": default_threshold,
        # Metric 1
        "metric_1_rejection_specificity_tnr": tnr_seq,
        "correct_rejections_count": N - seq_false_merges,
        # Metric 2
        "metric_2_false_merge_rate_far": far_seq,
        "false_merges_count": seq_false_merges,
        # Metric 3
        "metric_3_discovered_clusters": discovered_clusters,
        "ideal_clusters": N,
        # Metric 4
        "metric_4_mu_impostor": mu_imp,
        "metric_4_median_impostor": median_imp,
        "metric_4_std_impostor": std_imp,
        "metric_4_worst_case_impostor": worst_case_imp,
        "metric_4_min_impostor": min_imp,
        # Metric 5
        "metric_5_safe_zero_fp_threshold": safe_zero_fp_threshold,
        # Detailed Logs & Curves
        "cumulative_clusters": cumulative_clusters,
        "max_impostor_scores": max_impostor_scores.tolist(),
        "threshold_sweep": threshold_sweep.tolist(),
        "specificity_curve": specificity_curve,
        "sequential_decisions": seq_decisions,
        "all_pairs_log": all_pairs_log
    }


def plot_pure_unique_benchmark(results_dict, output_path):

    fig = plt.figure(figsize=(18, 5.5), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    colors = {
        "Siamese_ConvNeXt_RGB": "#2563eb",         # Royal Blue
        "Siamese_ConvNeXt_GRAY": "#0d9488",        # Teal
        "Siamese_ConvNeXt": "#2563eb",             # Default Blue
        "BiModel_Consensus": "#059669"             # Emerald Green
    }
    display_names = {
        "Siamese_ConvNeXt_RGB": "Siamese RGB (ConvNeXt)",
        "Siamese_ConvNeXt_GRAY": "Siamese Grayscale (ConvNeXt)",
        "Siamese_ConvNeXt": "Siamese (ConvNeXt-Tiny)",
        "BiModel_Consensus": "Bi-Model Consensus (AKAZE+SIFT)"
    }

    # -------------------------------------------------------------------------
    # Panel A: Discovered Clusters vs Image Index
    # -------------------------------------------------------------------------
    first_res = list(results_dict.values())[0]
    N = first_res["total_unique_toads"]
    x_indices = np.arange(1, N + 1)

    # Ideal diagonal
    ax1.plot(x_indices, x_indices, 'k--', lw=2.0, alpha=0.7, label=f"Ground Truth Ideal ({N} Unique Clusters)")

    for name, res in results_dict.items():
        c = colors.get(name, "#64748b")
        ax1.plot(
            x_indices, res["cumulative_clusters"],
            lw=2.5, color=c,
            label=f"{display_names.get(name, name)}: {res['metric_3_discovered_clusters']}/{N} Clusters"
        )
        ax1.scatter(N, res["metric_3_discovered_clusters"], color=c, s=55, zorder=5)

    ax1.set_title("A) Metric 3: Online Unique Toad Cluster Discovery", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Processed Images (Sequential Order)", fontsize=10, fontweight="600")
    ax1.set_ylabel("Discovered Unique Toad Clusters", fontsize=10, fontweight="600")
    ax1.set_xlim(1, N)
    ax1.set_ylim(0, N + 5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel B: Max Impostor Score Boxplot Distribution
    # -------------------------------------------------------------------------
    score_data = []
    labels = []
    box_colors = []
    for name, res in results_dict.items():
        score_data.append(res["max_impostor_scores"])
        labels.append(display_names.get(name, name))
        box_colors.append(colors.get(name, "#64748b"))

    bp = ax2.boxplot(score_data, tick_labels=labels, patch_artist=True, widths=0.48)
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    for idx, (name, res) in enumerate(results_dict.items()):
        m_val = res["metric_4_mu_impostor"]
        max_v = res["metric_4_worst_case_impostor"]
        t_def = res["operating_threshold"]
        ax2.scatter(idx + 1, m_val, color="red", s=35, zorder=5, label="Mean Max Impostor (μ)" if idx == 0 else "")
        ax2.annotate(f"Max: {max_v:.3f}\nThresh: {t_def:.3f}", (idx + 1, max_v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, fontweight='bold')

    ax2.set_title("B) Metric 4: Max-Impostor Score Distribution", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Maximum Impostor Score", fontsize=10, fontweight="600")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel C: Rejection Specificity (TNR %) vs Threshold
    # -------------------------------------------------------------------------
    for name, res in results_dict.items():
        c = colors.get(name, "#64748b")
        t_sweep = np.array(res["threshold_sweep"])
        ax3.plot(
            t_sweep, res["specificity_curve"],
            lw=2.4, color=c,
            label=f"{display_names.get(name, name)} (0 FP at τ ≥ {res['metric_5_safe_zero_fp_threshold']:.3f})"
        )
        t_def = res["operating_threshold"]
        idx_near = np.argmin(np.abs(t_sweep - t_def))
        tnr_def = res["specificity_curve"][idx_near]
        ax3.scatter(t_def, tnr_def, color=c, s=50, zorder=5)
        ax3.annotate(f"{tnr_def:.1f}% @ τ={t_def}", (t_def, tnr_def), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=7.5, fontweight='bold', color=c)

    ax3.set_title("C) Metrics 1 & 5: Rejection Rate & Safe Threshold", fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("Decision Threshold (τ)", fontsize=10, fontweight="600")
    ax3.set_ylabel("Rejection Specificity / TNR (%)", fontsize=10, fontweight="600")
    ax3.set_ylim(0, 105)
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved pure unique rejection plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Strict Unique Toad Rejection & False Merge Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=["siamese", "consensus", "both"],
        default="both",
        help="Model to test: 'siamese', 'consensus', or 'both' (default: both)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rgb", "gray"],
        default="rgb",
        help="Image mode for Siamese: 'rgb' or 'gray' (default: rgb)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to unique toads folder (default: 03_2_Siamese_network/data/unique_toads/{rgb_strict_unique,gray_strict_unique})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: results/evaluation)"
    )
    parser.add_argument(
        "--siamese_threshold",
        type=float,
        default=0.70,
        help="Operating similarity threshold for Siamese ConvNeXt (default: 0.70)"
    )
    parser.add_argument(
        "--consensus_threshold",
        type=float,
        default=0.060,
        help="Operating consensus threshold for Bi-Model Consensus (default: 0.060)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom model checkpoint path"
    )
    args = parser.parse_args()

    siamese_root = project_root / "03_2_Siamese_network"
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        unique_folder = "gray_strict_unique" if args.mode == "gray" else "rgb_strict_unique"
        data_dir = siamese_root / "data" / "unique_toads" / unique_folder
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "evaluation"
    plots_dir = project_root / "results" / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    is_tb = "01-02_Throat_Belly_pipeline" in str(data_dir) or (args.checkpoint and "01-02_Throat_Belly_pipeline" in str(args.checkpoint))
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if is_tb:
            checkpoint_path = siamese_root / "checkpoints" / f"{args.mode}_throat_belly" / "best_siamese_model.pt"
        else:
            checkpoint_path = siamese_root / "checkpoints" / args.mode / "best_siamese_model.pt"

    files, paths = load_unique_toad_images(data_dir)

    print("=" * 88)
    print("🐸 5 METRICS STRICTLY APPLICABLE TO ONLY UNIQUE TOADS (SEQUENTIAL ONLINE STREAM)")
    print(f"📂 Dataset:        {data_dir} ({len(files)} human-verified strictly unique toads)")
    print(f"🎯 Objective:      Verify model ability to REJECT false matches & create exactly {len(files)} unique IDs")
    print(f"⚙️ Model Choice:   [{args.model.upper()}] (Mode: {args.mode.upper()})")
    print(f"📏 Operating Thresh: Siamese τ={args.siamese_threshold} | Consensus T*={args.consensus_threshold}")
    print("=" * 88)

    results = {}

    # 1. Siamese Network
    if args.model in ["siamese", "both"]:
        sim_siamese, t_s = compute_siamese_similarity_matrix(paths, siamese_root, mode=args.mode, checkpoint_path=checkpoint_path)
        model_name = f"Siamese_ConvNeXt_{args.mode.upper()}_Throat_Belly" if is_tb else f"Siamese_ConvNeXt_{args.mode.upper()}"
        res_siamese = evaluate_pure_unique_metrics(sim_siamese, files, model_name, args.siamese_threshold)
        res_siamese["runtime_seconds"] = t_s
        results[model_name] = res_siamese

    # 2. Bi-Model Consensus
    if args.model in ["consensus", "both"]:
        sim_consensus, t_c = compute_consensus_similarity_matrix(paths)
        res_consensus = evaluate_pure_unique_metrics(sim_consensus, files, "BiModel_Consensus", args.consensus_threshold)
        res_consensus["runtime_seconds"] = t_c
        results["BiModel_Consensus"] = res_consensus

    # =========================================================================
    # Print the 5 Pure-Unique Metrics Table
    # =========================================================================
    print("\n" + "=" * 88)
    print(f"🏆 THE 5 METRICS FOR STRICTLY UNIQUE TOADS (N={len(files)} Unique Toads)")
    print("=" * 88)
    header = f"{'Unique Toad Metric':<35} | " + " | ".join([f"{name:<24}" for name in results.keys()])
    print(header)
    print("-" * 88)

    five_metric_rows = [
        ("1. Rejection Specificity (TNR %)", lambda r: f"{r['metric_1_rejection_specificity_tnr']:.2f}% ({r['correct_rejections_count']}/{r['total_unique_toads']})"),
        ("2. False Merge Rate (FAR %)", lambda r: f"{r['metric_2_false_merge_rate_far']:.2f}% ({r['false_merges_count']}/{r['total_unique_toads']})"),
        ("3. Discovered Unique Clusters", lambda r: f"{r['metric_3_discovered_clusters']} / {r['ideal_clusters']} (Ideal=60)"),
        ("4. Max Impostor Score (μ ± σ)", lambda r: f"{r['metric_4_mu_impostor']:.4f} ± {r['metric_4_std_impostor']:.4f}"),
        ("   — Median Max-Impostor Score", lambda r: f"{r['metric_4_median_impostor']:.4f}"),
        ("   — Worst-Case Impostor Score", lambda r: f"{r['metric_4_worst_case_impostor']:.4f}"),
        ("5. Safe Zero-FP Threshold (τ_safe)", lambda r: f"τ ≥ {r['metric_5_safe_zero_fp_threshold']:.4f}"),
        ("   Operating Threshold (τ)", lambda r: f"τ = {r['operating_threshold']:.3f}"),
        ("   Runtime", lambda r: f"{r['runtime_seconds']:.2f} s")
    ]

    for label, fn in five_metric_rows:
        row_str = f"{label:<35} | "
        vals = [f"{fn(r):<24}" for r in results.values()]
        print(row_str + " | ".join(vals))
    print("=" * 88)

    # =========================================================================
    # Generate Plots
    # =========================================================================
    plot_p = plots_dir / "strict_unique_rejection_benchmark.png"
    plot_pure_unique_benchmark(results, plot_p)

    # =========================================================================
    # Export Multi-Sheet Excel Workbook
    # =========================================================================
    excel_path = output_dir / "Strict_Unique_Rejection_Benchmark.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Summary 5 Metrics Table
        summary_rows = []
        for name, r in results.items():
            summary_rows.append({
                "Model": name,
                "Total Unique Toads": r["total_unique_toads"],
                "1. Rejection Specificity (TNR %)": f"{r['metric_1_rejection_specificity_tnr']:.2f}%",
                "2. False Merge Rate (FAR %)": f"{r['metric_2_false_merge_rate_far']:.2f}%",
                "3. Discovered Unique Clusters": f"{r['metric_3_discovered_clusters']} / {r['ideal_clusters']}",
                "4. Mean Max-Impostor Score (μ)": r["metric_4_mu_impostor"],
                "   Median Max-Impostor Score": r["metric_4_median_impostor"],
                "   Worst-Case Impostor Score": r["metric_4_worst_case_impostor"],
                "5. Safe Zero-FP Threshold (τ_safe)": r["metric_5_safe_zero_fp_threshold"],
                "Operating Threshold (τ)": r["operating_threshold"],
                "Runtime (s)": f"{r['runtime_seconds']:.2f}"
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="5_Unique_Metrics_Summary", index=False)

        # Sheet 2: Sequential Decisions Log
        for name, r in results.items():
            sheet_title = f"{name[:15]}_Seq_Log"
            pd.DataFrame(r["sequential_decisions"]).to_excel(writer, sheet_name=sheet_title, index=False)

        # Sheet 3: All-Pairs Max Impostors Log
        for name, r in results.items():
            sheet_title = f"{name[:15]}_AllPairs"
            pd.DataFrame(r["all_pairs_log"]).to_excel(writer, sheet_name=sheet_title, index=False)

    # =========================================================================
    # Export JSON Manifest
    # =========================================================================
    json_path = output_dir / "Strict_Unique_Rejection_Benchmark.json"
    benchmark_json = {
        "dataset_summary": {
            "source_directory": str(data_dir),
            "total_unique_toads": len(files),
            "ground_truth_property": "All 60 images belong to mutually disjoint unique toad identities."
        },
        "five_unique_toad_metrics": {
            name: {
                "metric_1_rejection_specificity_tnr": r["metric_1_rejection_specificity_tnr"],
                "metric_2_false_merge_rate_far": r["metric_2_false_merge_rate_far"],
                "metric_3_discovered_clusters": r["metric_3_discovered_clusters"],
                "metric_3_ideal_clusters": r["ideal_clusters"],
                "metric_4_mu_impostor": r["metric_4_mu_impostor"],
                "metric_4_median_impostor": r["metric_4_median_impostor"],
                "metric_4_worst_case_impostor": r["metric_4_worst_case_impostor"],
                "metric_5_safe_zero_fp_threshold": r["metric_5_safe_zero_fp_threshold"],
                "operating_threshold": r["operating_threshold"],
                "runtime_seconds": r["runtime_seconds"]
            }
            for name, r in results.items()
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_json, f, indent=2, ensure_ascii=False)

    print(f"📄 Saved detailed Excel workbook to: {excel_path}")
    print(f"📄 Saved benchmark JSON summary to:  {json_path}")
    print("=" * 88)
    print("🎉 PURE UNIQUE TOAD BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 88)


if __name__ == "__main__":
    main()

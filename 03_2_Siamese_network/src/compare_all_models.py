#!/usr/bin/env python3

# Comprehensive Benchmark Comparison: Siamese Network vs Bi-Model Consensus vs WildID on throat alone


import os
import sys
import json
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.0

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(current_dir))

from run_test_inference import run_siamese_inference, run_consensus_inference, load_test_images


def load_ground_truth(manifest_path: Path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    toad_clusters = manifest.get('clusters', {})
    img_to_toad = {}
    for t_id, imgs in toad_clusters.items():
        for img in imgs:
            norm_name = unicodedata.normalize('NFC', img)
            img_to_toad[norm_name] = t_id

    return toad_clusters, img_to_toad, manifest


def load_wildid_results(wildid_file: Path):
    wildid_map = {}
    if not wildid_file or not wildid_file.exists():
        return wildid_map

    with open(wildid_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 5:
                q_img = unicodedata.normalize('NFC', parts[1].strip())
                cand_img = unicodedata.normalize('NFC', parts[3].strip())
                try:
                    rank = int(parts[4].strip())
                except ValueError:
                    rank = 0
                wildid_map[q_img] = {
                    "matched_candidate": cand_img,
                    "rank": rank
                }
    return wildid_map


def evaluate_ranked_predictions(predictions, test_files, img_to_toad, model_name, top_k=20):

    total_queries = len(test_files)
    seen_toads = set()
    
    recap_hits = {k: 0 for k in range(1, top_k + 1)}
    recap_aps = []
    recap_first_ranks = []
    query_details = []
    recapture_details = []

    for i, q_file in enumerate(test_files):
        q_norm = unicodedata.normalize('NFC', q_file)
        q_toad = img_to_toad.get(q_norm, "Unknown")
        is_recapture = (q_toad in seen_toads)
        seen_toads.add(q_toad)

        cand_list = predictions.get(q_file, [])
        matches_ranks = []
        is_pos_list = []

        for c_entry in cand_list:
            c_file = unicodedata.normalize('NFC', c_entry["candidate"])
            c_toad = img_to_toad.get(c_file, "Unknown")
            c_rank = c_entry["rank"]

            # Identity-level match: any photo belonging to the same toad ID
            is_match = (c_toad == q_toad)
            is_pos_list.append(is_match)
            if is_match:
                matches_ranks.append(c_rank)

        first_rank = matches_ranks[0] if matches_ranks else 999
        top1_entry = cand_list[0] if cand_list else {}
        top1_file = top1_entry.get("candidate", "NONE")
        top1_toad = img_to_toad.get(unicodedata.normalize('NFC', top1_file), "NONE") if top1_file != "NONE" else "NONE"
        top1_score = top1_entry.get("score", 0.0)

        # Calculate Average Precision (AP) for recaptures
        if matches_ranks:
            cum_hits = 0
            prec_sum = 0.0
            for r_idx, is_pos in enumerate(is_pos_list):
                if is_pos:
                    cum_hits += 1
                    prec_sum += cum_hits / (r_idx + 1)
            ap = prec_sum / len(matches_ranks)
        else:
            ap = 0.0

        if is_recapture:
            recap_first_ranks.append(first_rank)
            recap_aps.append(ap)
            for k in range(1, top_k + 1):
                if first_rank <= k:
                    recap_hits[k] += 1

            recapture_details.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Ground_Truth_Toad_ID": q_toad,
                f"{model_name}_First_Match_Rank": first_rank,
                f"{model_name}_Top1_Candidate": top1_file,
                f"{model_name}_Top1_Toad_ID": top1_toad,
                f"{model_name}_Top1_Correct": (top1_toad == q_toad),
                f"{model_name}_Top1_Score": top1_score,
                f"{model_name}_In_Top3": (first_rank <= 3),
                f"{model_name}_In_Top5": (first_rank <= 5),
                f"{model_name}_In_Top10": (first_rank <= 10),
                f"{model_name}_In_Top20": (first_rank <= 20),
                f"{model_name}_AP": ap
            })

        query_details.append({
            "Query_Index": i + 1,
            "Query_Image": q_file,
            "Ground_Truth_Toad_ID": q_toad,
            "Encounter_Type": "Recapture" if is_recapture else "Initial_Sighting",
            f"{model_name}_First_Match_Rank": first_rank if is_recapture else 0,
            f"{model_name}_Top1_Candidate": top1_file,
            f"{model_name}_Top1_Toad_ID": top1_toad,
            f"{model_name}_Top1_Correct": (top1_toad == q_toad) if is_recapture else False,
            f"{model_name}_Top1_Score": top1_score,
            f"{model_name}_AP": ap if is_recapture else 0.0
        })

    num_recaptures = len(recap_first_ranks)
    cmc_curve = [(recap_hits[k] / num_recaptures) * 100.0 for k in range(1, top_k + 1)] if num_recaptures > 0 else [0.0] * top_k
    mean_ap = float(np.mean(recap_aps) * 100.0) if recap_aps else 0.0
    valid_ranks = [r for r in recap_first_ranks if r <= top_k]
    mean_first_rank = float(np.mean(valid_ranks)) if valid_ranks else float(top_k)

    return {
        "model_name": model_name,
        "total_queries": total_queries,
        "num_recaptures": num_recaptures,
        "num_initial_sightings": total_queries - num_recaptures,
        "cmc_curve": cmc_curve,
        "top1": cmc_curve[0],
        "top3": cmc_curve[2],
        "top5": cmc_curve[4],
        "top10": cmc_curve[9],
        "top20": cmc_curve[19],
        "mAP": mean_ap,
        "mean_first_rank": mean_first_rank,
        "recap_first_ranks": recap_first_ranks,
        "recapture_details": recapture_details,
        "query_details": query_details
    }


def evaluate_wildid(wildid_map, test_files, img_to_toad):
    if not wildid_map:
        return None

    total_queries = len(test_files)
    seen_toads = set()
    recap_hits = {k: 0 for k in range(1, 21)}
    recap_first_ranks = []
    query_details = []
    recapture_details = []

    for i, q_file in enumerate(test_files):
        q_norm = unicodedata.normalize('NFC', q_file)
        q_toad = img_to_toad.get(q_norm, "Unknown")
        is_recapture = (q_toad in seen_toads)
        seen_toads.add(q_toad)

        w_entry = wildid_map.get(q_norm, None)
        if w_entry and w_entry["matched_candidate"] != "NONE":
            cand_norm = unicodedata.normalize('NFC', w_entry["matched_candidate"])
            cand_toad = img_to_toad.get(cand_norm, "Unknown")
            is_correct = (cand_toad == q_toad)
            rank = w_entry["rank"] if is_correct else 999
        else:
            rank = 999
            is_correct = False
            cand_norm = "NONE"
            cand_toad = "NONE"

        if is_recapture:
            recap_first_ranks.append(rank)
            for k in range(1, 21):
                if rank <= k:
                    recap_hits[k] += 1

            recapture_details.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Ground_Truth_Toad_ID": q_toad,
                "WildID_First_Match_Rank": rank,
                "WildID_Matched_Candidate": cand_norm,
                "WildID_Candidate_Toad_ID": cand_toad,
                "WildID_Correct": is_correct,
                "WildID_In_Top3": (rank <= 3),
                "WildID_In_Top5": (rank <= 5),
                "WildID_In_Top10": (rank <= 10),
                "WildID_In_Top20": (rank <= 20)
            })

        query_details.append({
            "Query_Index": i + 1,
            "Query_Image": q_file,
            "Ground_Truth_Toad_ID": q_toad,
            "Encounter_Type": "Recapture" if is_recapture else "Initial_Sighting",
            "WildID_First_Match_Rank": rank if is_recapture else 0,
            "WildID_Matched_Candidate": cand_norm,
            "WildID_Candidate_Toad_ID": cand_toad,
            "WildID_Correct": is_correct if is_recapture else False
        })

    num_recaptures = len(recap_first_ranks)
    cmc_curve = [(recap_hits[k] / num_recaptures) * 100.0 for k in range(1, 21)] if num_recaptures > 0 else [0.0] * 20
    valid_ranks = [r for r in recap_first_ranks if r <= 20]
    mean_first_rank = float(np.mean(valid_ranks)) if valid_ranks else 20.0

    return {
        "model_name": "WildID",
        "total_queries": total_queries,
        "num_recaptures": num_recaptures,
        "num_initial_sightings": total_queries - num_recaptures,
        "cmc_curve": cmc_curve,
        "top1": cmc_curve[0],
        "top3": cmc_curve[2],
        "top5": cmc_curve[4],
        "top10": cmc_curve[9],
        "top20": cmc_curve[19],
        "mAP": cmc_curve[0],  # Single match baseline approximation
        "mean_first_rank": mean_first_rank,
        "recap_first_ranks": recap_first_ranks,
        "recapture_details": recapture_details,
        "query_details": query_details
    }


def plot_comprehensive_benchmark(model_results, output_path):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    ranks_x = np.arange(1, 21)

    colors = {
        "Siamese_ConvNeXt": "#2563eb",       # Royal Blue
        "BiModel_Consensus": "#059669",      # Emerald Green
        "WildID": "#d97706"                  # Amber / Orange
    }

    display_names = {
        "Siamese_ConvNeXt": "Siamese (ConvNeXt-Tiny)",
        "BiModel_Consensus": "Bi-Model Consensus (AKAZE+SIFT)",
        "WildID": "WildID"
    }

    markers = {
        "Siamese_ConvNeXt": 'o',
        "BiModel_Consensus": 's',
        "WildID": '^'
    }

    # -------------------------------------------------------------------------
    # CMC Recall Curves on Recaptures (Top-1 to Top-20)
    # -------------------------------------------------------------------------
    for name, res in model_results.items():
        c = colors.get(name, "#64748b")
        m = markers.get(name, 'o')
        lbl = display_names.get(name, name)
        ax.plot(
            ranks_x, res["cmc_curve"],
            marker=m, lw=2.4, markersize=6.0, color=c,
            label=lbl
        )

    ax.set_title("Cumulative Matching Characteristic (CMC)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rank Threshold (K)", fontsize=11, fontweight="600")
    ax.set_ylabel("Recapture Recall Rate (%)", fontsize=11, fontweight="600")
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.set_ylim(30, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved comparative plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Multi-Model Benchmark Comparison")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to test manifest JSON (default: 03_2_Siamese_network/data/test/test_manifest.json)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to flat test images folder (default: 03_2_Siamese_network/data/test/rgb_test)"
    )
    parser.add_argument(
        "--wildid_file",
        type=str,
        default=None,
        help="Path to WildID confirmed matches file (e.g. confirmed-matches.txt)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for reports & figures (default: results/evaluation)"
    )
    args = parser.parse_args()

    siamese_root = project_root / "03_2_Siamese_network"
    manifest_path = Path(args.manifest) if args.manifest else siamese_root / "data" / "test" / "test_manifest.json"
    data_dir = Path(args.data_dir) if args.data_dir else siamese_root / "data" / "test" / "rgb_test"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "evaluation"
    plots_dir = project_root / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("🔬 COMPREHENSIVE BIOMETRIC RE-ID BENCHMARK COMPARISON (SEQUENTIAL PROTOCOL)")
    print(f"📂 Test Data:  {data_dir}")
    print(f"📄 Manifest:   {manifest_path}")
    print("=" * 88)

    toad_clusters, img_to_toad, manifest_data = load_ground_truth(manifest_path)
    test_files, test_paths = load_test_images(data_dir)
    print(f"📊 Dataset: {len(test_files)} sequential test images across {len(toad_clusters)} ground-truth toad IDs.")
    print("🎯 Evaluation Protocol: Sequential growing gallery with descending score ranking (Top <= 20).\n")

    # 1. Run Siamese Network
    siamese_preds, _, _, t_s = run_siamese_inference(test_files, test_paths, siamese_root, top_k=20)
    res_siamese = evaluate_ranked_predictions(siamese_preds, test_files, img_to_toad, "Siamese_ConvNeXt")
    res_siamese["runtime"] = t_s

    # 2. Run Bi-Model Consensus
    consensus_preds, _, _, t_c = run_consensus_inference(test_files, test_paths, top_k=20)
    res_consensus = evaluate_ranked_predictions(consensus_preds, test_files, img_to_toad, "BiModel_Consensus")
    res_consensus["runtime"] = t_c

    model_results = {
        "Siamese_ConvNeXt": res_siamese,
        "BiModel_Consensus": res_consensus
    }

    # 3. Load WildID if provided or autodetected
    wildid_path = Path(args.wildid_file) if args.wildid_file else None
    if wildid_path is None:
        candidates = [
            siamese_root / "data" / "test" / "wildID" / "confirmed-matches.txt",
            siamese_root / "WildID" / "WildID_test_1" / "confirmed-matches.txt",
            siamese_root / "data" / "wildID" / "test" / "confirmed-matches.txt"
        ]
        for cand in candidates:
            if cand.exists():
                wildid_path = cand
                break

    if wildid_path and wildid_path.exists():
        print(f"🔍 Found WildID baseline results at: {wildid_path}")
        wildid_map = load_wildid_results(wildid_path)
        res_wildid = evaluate_wildid(wildid_map, test_files, img_to_toad)
        if res_wildid:
            model_results["WildID"] = res_wildid

    # =========================================================================
    # 4. PRINT COMPARATIVE SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 88)
    print(f"🏆 RECAPTURE IDENTIFICATION BENCHMARK SUMMARY (N={res_siamese['num_recaptures']} Recaptures / {res_siamese['total_queries']} Total Images)")
    print("=" * 88)
    header = f"{'Metric':<28} | " + " | ".join([f"{name:<22}" for name in model_results.keys()])
    print(header)
    print("-" * 88)
    for m_key, m_name in [
        ("top1", "Top-1 Accuracy"), ("top3", "Top-3 Accuracy"),
        ("top5", "Top-5 Accuracy"), ("top10", "Top-10 Accuracy"),
        ("top20", "Top-20 Accuracy"), ("mAP", "Mean Average Precision"),
        ("mean_first_rank", "Mean 1st-Match Rank")
    ]:
        row_str = f"{m_name:<28} | "
        vals = []
        for name, r in model_results.items():
            val = r[m_key]
            if "Accuracy" in m_name or "Precision" in m_name:
                vals.append(f"{val:>6.2f}%{'':<15}")
            else:
                vals.append(f"{val:>6.2f}{'':<16}")
        print(row_str + " | ".join(vals))
    print("=" * 88)

    # =========================================================================
    # 5. GENERATE AND SAVE PLOTS
    # =========================================================================
    plot_path = plots_dir / "Throat_model_comparison.png"
    plot_comprehensive_benchmark(model_results, plot_path)

    # =========================================================================
    # 6. EXPORT DETAILED EXCEL & JSON
    # =========================================================================
    excel_path = output_dir / "Comprehensive_Model_Benchmark.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Summary Table
        summary_rows = []
        for name, r in model_results.items():
            summary_rows.append({
                "Model": name,
                "Evaluated Recaptures": r["num_recaptures"],
                "Total Sequential Images": r["total_queries"],
                "Top-1 Accuracy": f"{r['top1']:.2f}%",
                "Top-3 Accuracy": f"{r['top3']:.2f}%",
                "Top-5 Accuracy": f"{r['top5']:.2f}%",
                "Top-10 Accuracy": f"{r['top10']:.2f}%",
                "Top-20 Accuracy": f"{r['top20']:.2f}%",
                "Mean Average Precision (mAP)": f"{r['mAP']:.2f}%",
                "Mean 1st-Match Rank": f"{r['mean_first_rank']:.2f}",
                "Runtime (s)": f"{r.get('runtime', 0):.2f}s"
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary_Benchmark", index=False)

        # Sheet 2: Recapture Side-by-Side Comparison
        df_recaps = []
        for name, r in model_results.items():
            df_recaps.append(pd.DataFrame(r["recapture_details"]))
        merged_recaps = df_recaps[0]
        for df_next in df_recaps[1:]:
            merged_recaps = pd.merge(merged_recaps, df_next, on=["Query_Index", "Query_Image", "Ground_Truth_Toad_ID"], how="outer")
        merged_recaps.to_excel(writer, sheet_name="Recapture_Rankings", index=False)

        # Sheet 3: All 100 Queries Log
        df_all = []
        for name, r in model_results.items():
            df_all.append(pd.DataFrame(r["query_details"]))
        merged_all = df_all[0]
        for df_next in df_all[1:]:
            merged_all = pd.merge(merged_all, df_next, on=["Query_Index", "Query_Image", "Ground_Truth_Toad_ID", "Encounter_Type"], how="outer")
        merged_all.to_excel(writer, sheet_name="All_100_Queries_Log", index=False)

    json_path = output_dir / "Comprehensive_Model_Benchmark.json"
    benchmark_json = {
        "dataset_metadata": manifest_data.get("dataset_summary", {}),
        "evaluation_summary": {
            "total_images": len(test_files),
            "unique_toads": len(toad_clusters),
            "initial_sightings": res_siamese["num_initial_sightings"],
            "recaptures": res_siamese["num_recaptures"]
        },
        "models": {
            name: {
                "top1": r["top1"], "top3": r["top3"], "top5": r["top5"],
                "top10": r["top10"], "top20": r["top20"], "mAP": r["mAP"],
                "mean_first_rank": r["mean_first_rank"],
                "runtime_seconds": r.get("runtime", 0),
                "cmc_curve": r["cmc_curve"]
            } for name, r in model_results.items()
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_json, f, indent=2, ensure_ascii=False)

    print(f"📄 Saved detailed Excel to: {excel_path}")
    print(f"📄 Saved benchmark JSON to: {json_path}")
    print("=" * 88)
    print("🎉 BENCHMARK COMPARISON COMPLETE!")
    print("=" * 88)


if __name__ == "__main__":
    main()

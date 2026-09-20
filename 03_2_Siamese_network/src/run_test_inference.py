#!/usr/bin/env python3

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


def load_test_images(data_dir: Path):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    test_files = sorted([
        unicodedata.normalize('NFC', f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(valid_exts) and not f.startswith('.')
    ])
    test_paths = [data_dir / f for f in test_files]
    return test_files, test_paths


def run_siamese_inference(test_files, test_paths, siamese_root: Path, mode: str = "rgb", top_k: int = 20, checkpoint_path: Path = None):

    print(f"⚡ Running Siamese Network (ConvNeXt-Tiny [{mode.upper()}]) inference...")
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
    for p in test_paths:
        e0, e180 = extract_siamese_embedding(p, model=model, transform=transform, return_tta=True)
        embs_0.append(e0)
        embs_180.append(e180)

    mat_0 = torch.stack(embs_0)
    mat_180 = torch.stack(embs_180)

    sim_0 = torch.mm(mat_0, mat_0.t()).numpy()
    sim_180 = torch.mm(mat_180, mat_0.t()).numpy()
    sim_matrix = np.maximum(sim_0, sim_180)

    N = len(test_files)
    predictions = {}
    csv_rows = []

    for i in range(N):
        q_file = test_files[i]
        
        if i == 0:
            # First image: no prior images in the gallery
            predictions[q_file] = []
            csv_rows.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Rank": 0,
                "Candidate_Image": "NONE",
                "Score": 0.0,
                "Similarity_Pct": 0.0
            })
            continue

        # Past candidates strictly from 0 to i-1
        past_indices = list(range(i))
        # Sort past candidates by similarity score in descending order
        ranked_past_indices = sorted(past_indices, key=lambda j: sim_matrix[i, j], reverse=True)[:top_k]
        
        cand_list = []
        for rank_idx, cand_idx in enumerate(ranked_past_indices):
            c_file = test_files[cand_idx]
            score = float(sim_matrix[i, cand_idx])
            sim_pct = max(0.0, min(100.0, ((score + 1.0) / 2.0) * 100.0))

            cand_list.append({
                "rank": rank_idx + 1,
                "candidate": c_file,
                "candidate_index": cand_idx + 1,
                "score": score,
                "similarity_pct": sim_pct
            })

            csv_rows.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Rank": rank_idx + 1,
                "Candidate_Image": c_file,
                "Candidate_Index": cand_idx + 1,
                "Score": score,
                "Similarity_Pct": sim_pct
            })

        predictions[q_file] = cand_list

    elapsed = time.time() - t0
    print(f"✅ Siamese inference completed for {N} images in {elapsed:.2f}s")
    return predictions, pd.DataFrame(csv_rows), sim_matrix, elapsed


def run_consensus_inference(test_files, test_paths, top_k: int = 20):
    print("⚡ Running Bi-Model Consensus Engine (AKAZE + SIFT) inference...")
    t0 = time.time()
    engine = BiModelConsensusEngine(akaze_weight=0.55, sift_weight=0.45, threshold_t_star=0.080)
    batch_data = engine.precompute_batch_matrices(test_paths)
    sim_matrix = batch_data["R_consensus"]

    N = len(test_files)
    predictions = {}
    csv_rows = []

    for i in range(N):
        q_file = test_files[i]
        
        if i == 0:
            # First image: no prior images in the gallery
            predictions[q_file] = []
            csv_rows.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Rank": 0,
                "Candidate_Image": "NONE",
                "Score": 0.0,
                "Similarity_Pct": 0.0
            })
            continue

        # Past candidates strictly from 0 to i-1
        past_indices = list(range(i))
        # Sort past candidates by consensus score in descending order
        ranked_past_indices = sorted(past_indices, key=lambda j: sim_matrix[i, j], reverse=True)[:top_k]
        
        cand_list = []
        for rank_idx, cand_idx in enumerate(ranked_past_indices):
            c_file = test_files[cand_idx]
            score = float(sim_matrix[i, cand_idx])
            sim_pct = max(0.0, min(100.0, score * 100.0))

            cand_list.append({
                "rank": rank_idx + 1,
                "candidate": c_file,
                "candidate_index": cand_idx + 1,
                "score": score,
                "similarity_pct": sim_pct
            })

            csv_rows.append({
                "Query_Index": i + 1,
                "Query_Image": q_file,
                "Rank": rank_idx + 1,
                "Candidate_Image": c_file,
                "Candidate_Index": cand_idx + 1,
                "Score": score,
                "Similarity_Pct": sim_pct
            })

        predictions[q_file] = cand_list

    elapsed = time.time() - t0
    print(f"✅ Bi-Model Consensus inference completed for {N} images in {elapsed:.2f}s")
    return predictions, pd.DataFrame(csv_rows), sim_matrix, elapsed


def main():
    parser = argparse.ArgumentParser(description="Run test inference for Siamese and/or Bi-Model Consensus")
    parser.add_argument(
        "--model",
        type=str,
        choices=["siamese", "consensus", "both"],
        default="both",
        help="Model to run: 'siamese', 'consensus', or 'both' (default: both)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rgb", "gray"],
        default="rgb",
        help="Siamese image mode: 'rgb' or 'gray' (default: rgb)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to flat test images folder (default: 03_2_Siamese_network/data/test/{rgb_test,gray_test})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for predictions (default: results/predictions)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of candidates to rank (default: 20)"
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
        test_folder = "gray_test" if args.mode == "gray" else "rgb_test"
        data_dir = siamese_root / "data" / "test" / test_folder
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    is_tb = "01-02_Throat_Belly_pipeline" in str(data_dir) or (args.checkpoint and "01-02_Throat_Belly_pipeline" in str(args.checkpoint))
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if is_tb:
            checkpoint_path = siamese_root / "checkpoints" / f"{args.mode}_throat_belly" / "best_siamese_model.pt"
        else:
            checkpoint_path = siamese_root / "checkpoints" / args.mode / "best_siamese_model.pt"

    test_files, test_paths = load_test_images(data_dir)
    print("=" * 75)
    print(f"🔬 RUNNING TEST INFERENCE ON: {data_dir.name} ({len(test_files)} images)")
    print(f"🎯 Model Selection: [{args.model.upper()}] (Mode: {args.mode.upper()}) | Top-K Candidates: {args.top_k}")
    print(f"📁 Output Directory: {output_dir}")
    print("=" * 75)

    prefix = f"siamese_{args.mode}_throat_belly" if is_tb else (f"siamese_{args.mode}" if args.mode == "gray" else "siamese")

    siam_out_dir = output_dir if output_dir.name == "siamese" else output_dir / "siamese"
    cons_out_dir = output_dir if output_dir.name == "consensus" else output_dir / "consensus"
    siam_out_dir.mkdir(parents=True, exist_ok=True)
    cons_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Siamese Network
    if args.model in ["siamese", "both"]:
        siamese_preds, df_siamese, mat_siamese, t_s = run_siamese_inference(test_files, test_paths, siamese_root, mode=args.mode, top_k=args.top_k, checkpoint_path=checkpoint_path)
        s_json = siam_out_dir / f"{prefix}_predictions.json"
        s_csv = siam_out_dir / f"{prefix}_rankings.csv"
        s_npz = siam_out_dir / f"{prefix}_similarity_matrix.npy"

        with open(s_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": f"Siamese_ConvNeXt_Tiny_{args.mode.upper()}",
                "total_queries": len(test_files),
                "top_k": args.top_k,
                "runtime_seconds": t_s,
                "predictions": siamese_preds
            }, f, indent=2, ensure_ascii=False)
        df_siamese.to_csv(s_csv, index=False)
        np.save(s_npz, mat_siamese)
        print(f"📄 Saved Siamese predictions to: {s_json}")
        print(f"📊 Saved Siamese rankings to:    {s_csv}")

    # 2. Bi-Model Consensus
    if args.model in ["consensus", "both"]:
        cons_preds, df_cons, mat_cons, t_c = run_consensus_inference(test_files, test_paths, top_k=args.top_k)
        c_json = cons_out_dir / "consensus_predictions.json"
        c_csv = cons_out_dir / "consensus_rankings.csv"
        c_npz = cons_out_dir / "consensus_similarity_matrix.npy"

        with open(c_json, "w", encoding="utf-8") as f:
            json.dump({
                "model": "BiModel_Consensus_AKAZE_SIFT",
                "total_queries": len(test_files),
                "top_k": args.top_k,
                "runtime_seconds": t_c,
                "predictions": cons_preds
            }, f, indent=2, ensure_ascii=False)
        df_cons.to_csv(c_csv, index=False)
        np.save(c_npz, mat_cons)
        print(f"📄 Saved Consensus predictions to: {c_json}")
        print(f"📊 Saved Consensus rankings to:    {c_csv}")

    print("=" * 75)
    print("🎉 Test inference completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pure Siamese Metric Retrieval Engine (Top-20)
==============================================
Extracts 256-d metric embeddings using the fine-tuned ConvNeXt-Tiny Siamese model.
Retrieves and ranks Top-20 candidates strictly based on embedding cosine similarity (with rotation TTA).
Completely decoupled from handcrafted feature matchers (no SIFT / AKAZE blending).
"""

import os
import sys
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add project root and local src to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(current_dir))

from model import ToadMetricEmbeddingNet
from dataset import get_default_transforms

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

_loaded_model = None
_gallery_cache = {}


def parse_filename_metadata(filename):
    """Extracts site, tag, index, and sex from filename (e.g. Aulbachtal_2025_Tag_03_A01_03_05_M_roh.jpg)."""
    parts = filename.split('_')
    meta = {
        'site': 'Unknown',
        'tag': 'Unknown',
        'sex': 'Unknown',
        'id_code': filename
    }
    if 'Aulbachtal' in filename:
        meta['site'] = 'Aulbachtal'
    elif 'Hochfläche' in filename or 'Hochflaeche' in filename:
        meta['site'] = 'Hochfläche'

    for p in parts:
        if p.startswith('Tag'):
            meta['tag'] = p
        if p in ('M', 'W', 'SA', 'J'):
            sex_map = {'M': 'Male (M)', 'W': 'Female (W)', 'SA': 'Subadult (SA)', 'J': 'Juvenile (J)'}
            meta['sex'] = sex_map.get(p, p)
    return meta


def load_siamese_model(checkpoint_path=None, mode="rgb", backbone="convnext_tiny", embedding_dim=256):
    """
    Loads fine-tuned ConvNeXt-Tiny Siamese network weights.
    Falls back to pre-trained base model if checkpoint is not yet generated.
    """
    global _loaded_model
    if checkpoint_path is None:
        checkpoint_path = project_root / "03_2_Siamese_network" / "checkpoints" / mode / "best_siamese_model.pt"

    model = ToadMetricEmbeddingNet(backbone_name=backbone, embedding_dim=embedding_dim).to(device)

    if os.path.exists(str(checkpoint_path)):
        try:
            ckpt = torch.load(str(checkpoint_path), map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"✅ Loaded fine-tuned Siamese ({backbone}) weights from: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint {checkpoint_path}: {e}. Using base model.")
    else:
        print(f"ℹ️ Siamese checkpoint not found at {checkpoint_path}. Using base pre-trained model.")

    model.eval()
    _loaded_model = model
    return model


def extract_siamese_embedding(image_path_or_pil, model=None, transform=None, return_tta=True):
    """
    Extracts L2-normalized 256-d metric embedding vector for a toad image.
    If return_tta=True, returns tuple (emb_0deg, emb_180deg) for rotation-invariant matching.
    """
    if model is None:
        global _loaded_model
        if _loaded_model is None:
            model = load_siamese_model()
        else:
            model = _loaded_model

    if transform is None:
        transform = get_default_transforms(is_train=False)

    if isinstance(image_path_or_pil, (str, Path)):
        img = Image.open(str(image_path_or_pil)).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    t0 = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb0 = model(t0).squeeze(0).cpu()

    if not return_tta:
        return emb0

    t180 = transform(img.rotate(180)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb180 = model(t180).squeeze(0).cpu()

    return emb0, emb180


def build_or_load_batch_embeddings(image_paths, mode="rgb", force_recompute=False):
    """
    Computes and caches Siamese Multi-Orientation TTA metric embeddings for an arbitrary batch of images.
    """
    global _gallery_cache
    sorted_paths = sorted([str(p) for p in image_paths])
    cache_key = f"batch_{mode}_{len(sorted_paths)}_{hash(tuple(sorted_paths))}"

    if not force_recompute and cache_key in _gallery_cache:
        return _gallery_cache[cache_key]

    model = load_siamese_model(mode=mode)
    transform = get_default_transforms(is_train=False)

    embs_0_list = []
    embs_180_list = []
    valid_paths = []

    print(f"⚙️ Computing Siamese TTA embeddings for {len(sorted_paths)} images...")
    for p in sorted_paths:
        try:
            e0, e180 = extract_siamese_embedding(p, model=model, transform=transform, return_tta=True)
            embs_0_list.append(e0)
            embs_180_list.append(e180)
            valid_paths.append(str(p))
        except Exception:
            continue

    if not embs_0_list:
        raise ValueError("No valid image embeddings could be computed for the batch.")

    batch_matrix_0 = torch.stack(embs_0_list)
    batch_matrix_180 = torch.stack(embs_180_list)

    batch_data = {
        "matrix_0": batch_matrix_0,
        "matrix_180": batch_matrix_180,
        "paths": valid_paths,
        "filenames": [os.path.basename(p) for p in valid_paths]
    }

    _gallery_cache[cache_key] = batch_data
    return batch_data


def get_top_20_within_batch_matches(query_image_path, batch_image_paths, mode="rgb", top_k=20, current_query_index=None):
    """
    Pure Siamese Metric Retrieval (Strict Sequential WildID Protocol):
    Query image i is compared exclusively against previously cataloged images (0 .. i-1).
    Returns Top-20 candidates.
    """
    sorted_paths = sorted([str(p) for p in batch_image_paths])
    batch_data = build_or_load_batch_embeddings(sorted_paths, mode=mode)
    batch_matrix_0 = batch_data["matrix_0"]
    batch_paths = batch_data["paths"]
    batch_filenames = batch_data["filenames"]

    query_str = str(query_image_path)
    if query_str not in batch_paths:
        return []

    q_idx = batch_paths.index(query_str)
    effective_idx = current_query_index if current_query_index is not None else q_idx

    # Strictly sequential: compare only against past images (0 .. effective_idx - 1)
    if effective_idx == 0:
        return []

    past_matrix_0 = batch_matrix_0[:effective_idx]
    model = load_siamese_model(mode=mode)
    transform = get_default_transforms(is_train=False)
    q_emb0, q_emb180 = extract_siamese_embedding(query_image_path, model=model, transform=transform, return_tta=True)

    # Multi-Orientation TTA Cosine Similarities against past gallery
    sims_0 = torch.mv(past_matrix_0, q_emb0).numpy()
    sims_180 = torch.mv(past_matrix_0, q_emb180).numpy()
    similarities = np.maximum(sims_0, sims_180)

    ranked_past_indices = np.argsort(similarities)[::-1][:top_k]
    candidates = []

    for rank_idx, idx in enumerate(ranked_past_indices):
        cand_path = batch_paths[idx]
        raw_sim = float(similarities[idx])
        sim_percentage = max(0.0, min(100.0, ((raw_sim + 1.0) / 2.0) * 100.0))
        cand_filename = batch_filenames[idx]

        candidates.append({
            "rank": rank_idx + 1,
            "filename": cand_filename,
            "path": cand_path,
            "cosine_similarity": raw_sim,
            "similarity_pct": sim_percentage,
            "metadata": parse_filename_metadata(cand_filename)
        })

    return candidates


def precompute_all_batch_matches(batch_image_paths, mode="rgb", top_k=20, progress_callback=None):
    """
    Precomputes Top-20 candidate rankings strictly sequentially for ALL images in a batch (WildID Protocol).
    Query i is compared exclusively against images 0 ... i-1.
    Returns a dictionary: { query_filename: [cand1, cand2, ... cand20] }
    """
    sorted_paths = sorted([str(p) for p in batch_image_paths])
    if len(sorted_paths) <= 1:
        return {}

    # 1. Build or load Siamese embeddings (vectorized)
    batch_data = build_or_load_batch_embeddings(sorted_paths, mode=mode)
    m0 = batch_data["matrix_0"]
    m180 = batch_data["matrix_180"]
    paths = batch_data["paths"]
    filenames = batch_data["filenames"]
    n_imgs = len(paths)

    # 2. Vectorized Multi-Orientation Pairwise Similarity Matrix
    sim_0 = torch.mm(m0, m0.T).numpy()
    sim_180 = torch.mm(m180, m0.T).numpy()
    pairwise_sim = np.maximum(sim_0, sim_180)

    results = {}

    for i in range(n_imgs):
        q_fname = filenames[i]

        if i == 0:
            results[q_fname] = []
            if progress_callback is not None:
                progress_callback(1 / n_imgs)
            continue

        # Strict sequential past candidates 0 ... i-1
        past_indices = list(range(i))
        ranked_past = sorted(past_indices, key=lambda j: pairwise_sim[i, j], reverse=True)[:top_k]

        cands = []
        for rank_idx, idx in enumerate(ranked_past):
            raw_sim = float(pairwise_sim[i, idx])
            sim_pct = max(0.0, min(100.0, ((raw_sim + 1.0) / 2.0) * 100.0))
            c_fname = filenames[idx]
            c_path = paths[idx]

            cands.append({
                "rank": rank_idx + 1,
                "filename": c_fname,
                "path": c_path,
                "cosine_similarity": raw_sim,
                "similarity_pct": sim_pct,
                "metadata": parse_filename_metadata(c_fname)
            })

        results[q_fname] = cands

        if progress_callback is not None:
            progress_callback((i + 1) / n_imgs)

    return results


def get_top_10_within_batch_matches(query_image_path, batch_image_paths, mode="rgb", top_k=20, **kwargs):
    """Backward compatibility alias returning Top-20 sequential Siamese candidates."""
    return get_top_20_within_batch_matches(query_image_path, batch_image_paths, mode=mode, top_k=top_k, **kwargs)


def get_top_10_siamese_matches(query_image_path, gallery_dir, mode="rgb", top_k=20, **kwargs):
    """Backward compatibility alias returning Top-20 sequential Siamese candidates."""
    image_paths = []
    for split in ["train", "val", "test"]:
        split_dir = Path(gallery_dir) / split
        if split_dir.exists():
            image_paths.extend(glob.glob(str(split_dir / "*.jpg")) + glob.glob(str(split_dir / "*.jpeg")) + glob.glob(str(split_dir / "*.png")))
    return get_top_20_within_batch_matches(query_image_path, image_paths, mode=mode, top_k=top_k, **kwargs)

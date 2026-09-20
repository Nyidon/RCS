import os
import sys
from pathlib import Path
import numpy as np
import cv2

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_extractor import DualFeatureExtractor
from pairwise_matcher import PairwiseMatcher

_cached_engine = None

def parse_filename_metadata(filename):
    """Extracts site, tag, index, and sex from filename."""
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


class BiModelConsensusEngine:
    """
    Bi-Model Consensus Engine fusing AKAZE (MLDB) and SIFT (L2) Ratio Normalization.
    
    Provides:
      1. Calibrated Threshold Filtering (T*)
      2. Top-20 Candidate Retrieval Engine (Sequential WildID Mode, Dynamic Catalog Mode, Global Batch Mode)
      3. Keypoint Spot Correspondence Visualizer
    """
    def __init__(self, akaze_weight=0.55, sift_weight=0.45, threshold_t_star=0.080):
        self.akaze_weight = akaze_weight
        self.sift_weight = sift_weight
        self.threshold_t_star = threshold_t_star
        self.extractor = DualFeatureExtractor()
        self.pairwise_matcher = PairwiseMatcher(self.extractor)
        
        # In-memory batch cache
        self._batch_cache = {}

    def get_consensus_score(self, r_akaze, r_sift):
        """Computes weighted bi-model consensus ratio score."""
        return (self.akaze_weight * r_akaze) + (self.sift_weight * r_sift)

    def precompute_batch_matrices(self, image_paths, progress_callback=None):
        """
        Precomputes N x N AKAZE and SIFT ratio matrices and caches them in memory.
        """
        paths = sorted([str(p) for p in image_paths])
        cache_key = f"batch_{len(paths)}_{hash(tuple(paths))}"
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]

        R_akaze, R_sift, valid_paths = self.pairwise_matcher.compute_all_pairwise_ratios(
            paths, progress_callback=progress_callback
        )
        
        # Symmetrized matrices: S_ij^sym = min(R_ij, R_ji)
        R_akaze_sym = np.minimum(R_akaze, R_akaze.T)
        R_sift_sym = np.minimum(R_sift, R_sift.T)
        R_consensus = (self.akaze_weight * R_akaze_sym) + (self.sift_weight * R_sift_sym)

        batch_data = {
            "paths": valid_paths,
            "filenames": [os.path.basename(p) for p in valid_paths],
            "R_akaze": R_akaze,
            "R_sift": R_sift,
            "R_akaze_sym": R_akaze_sym,
            "R_sift_sym": R_sift_sym,
            "R_consensus": R_consensus
        }

        self._batch_cache[cache_key] = batch_data
        return batch_data

    def get_top_20_matches(
        self,
        query_image_path,
        batch_image_paths,
        mode="sequential",  # 'sequential' (WildID), 'catalog', or 'batch'
        top_k=20,
        catalog_state=None,  # dict mapping Toad_ID -> list of confirmed image paths
        current_query_index=None
    ):
        """
        Retrieves ranked Top-20 candidates for a query image under the chosen operational mode:
        
        1. 'sequential' (Exact WildID behavior):
           - Query at index i is compared ONLY against past images 0 .. i-1.
           - Returns min(i, top_k) candidate cards.
           - Image 0 returns [] (Always New Toad).
           
        2. 'catalog' (Dynamic Toad Catalog Mode):
           - Query is compared against the primary exemplar images of all currently confirmed Toad IDs.
           - Returns up to top_k unique toad candidates.
           
        3. 'batch' (Global Batch Mode):
           - Query is compared against all other images in the dataset.
        """
        paths = sorted([str(p) for p in batch_image_paths])
        batch_data = self.precompute_batch_matrices(paths)
        
        p_list = batch_data["paths"]
        f_list = batch_data["filenames"]
        R_cons = batch_data["R_consensus"]
        R_akaze = batch_data["R_akaze_sym"]
        R_sift = batch_data["R_sift_sym"]

        query_str = str(query_image_path)
        if query_str not in p_list:
            # Fallback if single new external query
            return []

        q_idx = p_list.index(query_str)
        if current_query_index is not None:
            effective_idx = current_query_index
        else:
            effective_idx = q_idx

        # Candidate selection based on mode
        if mode == "sequential":
            # WildID sequential: strictly compare against past images (0 .. effective_idx - 1)
            if effective_idx == 0:
                return []
            candidate_indices = list(range(effective_idx))
            
        elif mode == "catalog" and catalog_state is not None and len(catalog_state) > 0:
            # Catalog mode: compare against primary image of each confirmed toad
            candidate_indices = []
            for toad_id, c_paths in catalog_state.items():
                if c_paths:
                    prim_path = str(c_paths[0])
                    if prim_path in p_list and prim_path != query_str:
                        candidate_indices.append(p_list.index(prim_path))
            if not candidate_indices:
                return []
        else:
            # Global batch mode: all images except self
            candidate_indices = [idx for idx in range(len(p_list)) if idx != q_idx]

        if not candidate_indices:
            return []

        # Extract scores for candidate pool
        cand_scores = []
        for c_idx in candidate_indices:
            c_path = p_list[c_idx]
            c_fname = f_list[c_idx]
            
            score_cons = float(R_cons[q_idx, c_idx])
            score_a = float(R_akaze[q_idx, c_idx])
            score_s = float(R_sift[q_idx, c_idx])
            
            cand_scores.append({
                "path": c_path,
                "filename": c_fname,
                "consensus_score": score_cons,
                "consensus_pct": max(0.0, min(100.0, score_cons * 100.0)),
                "akaze_ratio": score_a,
                "akaze_ratio_pct": max(0.0, min(100.0, score_a * 100.0)),
                "sift_ratio": score_s,
                "sift_ratio_pct": max(0.0, min(100.0, score_s * 100.0)),
                "is_confident_match": score_cons >= self.threshold_t_star,
                "metadata": parse_filename_metadata(c_fname)
            })

        # Sort descending by consensus ratio score
        cand_scores.sort(key=lambda x: x["consensus_score"], reverse=True)
        
        # Take Top-K
        top_k_candidates = cand_scores[:top_k]
        for r, cand in enumerate(top_k_candidates):
            cand["rank"] = r + 1

        return top_k_candidates

    def generate_spot_matching_visualization(self, query_img_path, cand_img_path, model_type="akaze"):
        """
        Generates side-by-side spot correspondence visualization image with inlier lines.
        """
        pair_data = self.extractor.compute_pair_similarity(query_img_path, cand_img_path)
        
        img1 = cv2.imread(str(query_img_path))
        img2 = cv2.imread(str(cand_img_path))
        if img1 is None or img2 is None:
            return None

        if model_type == "akaze":
            if pair_data['akaze_rot180']:
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            kpts1 = pair_data['akaze_kpts1']
            kpts2 = pair_data['akaze_kpts2']
            matches = pair_data['akaze_match_list'][:40]  # Show top 40 cleanest lines
        else:
            if pair_data['sift_rot180']:
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            kpts1 = pair_data['sift_kpts1']
            kpts2 = pair_data['sift_kpts2']
            matches = pair_data['sift_match_list'][:40]

        vis_img = cv2.drawMatches(
            img1, kpts1,
            img2, kpts2,
            matches, None,
            matchColor=(0, 255, 128),  # Vibrant mint green spot lines
            singlePointColor=(0, 100, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


def get_consensus_engine():
    global _cached_engine
    if _cached_engine is None:
        _cached_engine = BiModelConsensusEngine()
    return _cached_engine

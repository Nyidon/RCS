from concurrent.futures import ThreadPoolExecutor
import numpy as np
from feature_extractor import DualFeatureExtractor

class PairwiseMatcher:

    def __init__(self, extractor=None, max_workers=8):
        if extractor is None:
            self.extractor = DualFeatureExtractor()
        else:
            self.extractor = extractor
        self.max_workers = max_workers

    def precompute_batch_features(self, image_paths, progress_callback=None):

        feats_0 = {}
        feats_180 = {}
        self_sims = {}
        
        n = len(image_paths)
        for i, p in enumerate(image_paths):
            p_str = str(p)
            f0 = self.extractor.extract_features(p_str, rotate_180=False)
            f180 = self.extractor.extract_features(p_str, rotate_180=True)
            ss = self.extractor.get_self_similarity(p_str)
            
            feats_0[p_str] = f0
            feats_180[p_str] = f180
            self_sims[p_str] = ss
            
            if progress_callback is not None:
                progress_callback((i + 1) / n)
                
        return feats_0, feats_180, self_sims

    def compute_all_pairwise_ratios(self, image_paths, progress_callback=None):

        paths = sorted([str(p) for p in image_paths])
        N = len(paths)
        if N == 0:
            return np.zeros((0, 0)), np.zeros((0, 0)), []

        # 1. Precompute features
        feats_0, feats_180, self_sims = self.precompute_batch_features(paths)

        R_primary = np.zeros((N, N), dtype=np.float32)
        R_secondary = np.zeros((N, N), dtype=np.float32)

        def match_row(i):
            p_i = paths[i]
            f_i_0 = feats_0[p_i]
            f_i_180 = feats_180[p_i]
            s_ii = self_sims[p_i]

            row_prim = np.zeros(N, dtype=np.float32)
            row_sec = np.zeros(N, dtype=np.float32)

            row_prim[i] = 1.0
            row_sec[i] = 1.0

            d_si0 = f_i_0['sift'][1]
            d_si180 = f_i_180['sift'][1]
            d_bi0 = f_i_0['model_b'][1]
            d_bi180 = f_i_180['model_b'][1]

            ss_s = s_ii['sift']
            ss_b = s_ii['model_b']

            for j in range(N):
                if i == j:
                    continue
                p_j = paths[j]
                f_j_0 = feats_0[p_j]

                d_sj0 = f_j_0['sift'][1]
                d_bj0 = f_j_0['model_b'][1]

                # SIFT matching
                m_s0, _ = self.extractor.match_descriptors(d_si0, d_sj0, model_type="sift")
                m_s180, _ = self.extractor.match_descriptors(d_si180, d_sj0, model_type="sift")
                best_s = max(m_s0, m_s180)
                row_prim[j] = best_s / ss_s

                # Secondary (AKAZE/ORB) matching
                m_b0, _ = self.extractor.match_descriptors(d_bi0, d_bj0, model_type="model_b")
                m_b180, _ = self.extractor.match_descriptors(d_bi180, d_bj0, model_type="model_b")
                best_b = max(m_b0, m_b180)
                row_sec[j] = best_b / ss_b

            return i, row_prim, row_sec

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(match_row, range(N)))

        for i, row_p, row_s in results:
            R_primary[i] = row_p
            R_secondary[i] = row_s
            if progress_callback is not None:
                progress_callback((i + 1) / N)

        return R_primary, R_secondary, paths

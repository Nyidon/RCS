import cv2
import numpy as np
from pathlib import Path

class DualFeatureExtractor:
    
    def __init__(
        self,
        sift_contrast_thresh=0.012,
        sift_nfeatures=1500,
        orb_nfeatures=1500,
        min_border_dist=6,
        lowe_ratio=0.80,
        ransac_thresh=5.0
    ):
        self.min_border_dist = min_border_dist
        self.lowe_ratio = lowe_ratio
        self.ransac_thresh = ransac_thresh
        
        # 1. Primary Model: SIFT
        self.sift = cv2.SIFT_create(
            nfeatures=sift_nfeatures,
            nOctaveLayers=3,
            contrastThreshold=sift_contrast_thresh,
            edgeThreshold=10,
            sigma=1.2
        )
        self.bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
        
        # 2. Secondary Model: AKAZE (if available) or ORB
        self.has_akaze = False
        if hasattr(cv2, 'AKAZE_create'):
            try:
                self.model_b = cv2.AKAZE_create(
                    descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                    descriptor_size=0,
                    descriptor_channels=3,
                    threshold=0.0008,
                    diffusivity=cv2.KAZE_DIFF_PM_G2
                )
                self.has_akaze = True
            except Exception:
                self.has_akaze = False
                
        if not self.has_akaze:
            self.model_b = cv2.ORB_create(
                nfeatures=orb_nfeatures,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=10,
                patchSize=31
            )
            
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Feature caches
        self._cache_primary = {}
        self._cache_secondary = {}
        self._cache_self_sim = {}

    def _get_inner_mask(self, gray_img):
        _, mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
        k_size = self.min_border_dist * 2 + 1
        kernel = np.ones((k_size, k_size), np.uint8)
        inner_mask = cv2.erode(mask, kernel)
        return inner_mask

    def extract_features(self, img_input, rotate_180=False):

        if isinstance(img_input, (str, Path)):
            path_str = str(img_input)
            cache_key = (path_str, rotate_180)
            if cache_key in self._cache_primary and cache_key in self._cache_secondary:
                return {
                    'sift': self._cache_primary[cache_key],
                    'model_b': self._cache_secondary[cache_key],
                    'akaze': self._cache_secondary[cache_key]
                }
            gray = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not read image at: {path_str}")
        elif isinstance(img_input, np.ndarray):
            cache_key = None
            if len(img_input.shape) == 3:
                gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_input.copy()
        else:
            raise TypeError("Input must be a filepath (str/Path) or numpy image array.")

        if rotate_180:
            gray = cv2.rotate(gray, cv2.ROTATE_180)

        inner_mask = self._get_inner_mask(gray)

        # Primary (SIFT) extraction
        kpts_s, descs_s = self.sift.detectAndCompute(gray, inner_mask)
        sift_tuple = (kpts_s, descs_s, len(kpts_s) if kpts_s is not None else 0)

        # Secondary (AKAZE/ORB) extraction
        kpts_b, descs_b = self.model_b.detectAndCompute(gray, inner_mask)
        model_b_tuple = (kpts_b, descs_b, len(kpts_b) if kpts_b is not None else 0)

        if cache_key is not None:
            self._cache_primary[cache_key] = sift_tuple
            self._cache_secondary[cache_key] = model_b_tuple

        return {
            'sift': sift_tuple,
            'model_b': model_b_tuple,
            'akaze': model_b_tuple,
            'gray_img': gray
        }

    def match_descriptors(self, descs1, descs2, model_type="sift"):

        if descs1 is None or descs2 is None:
            return 0, []
        if len(descs1) < 2 or len(descs2) < 2:
            return 0, []

        matcher = self.bf_l2 if model_type == "sift" else self.bf_hamming

        try:
            raw_matches = matcher.knnMatch(descs1, descs2, k=2)
        except Exception:
            return 0, []

        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.lowe_ratio * n.distance:
                    good_matches.append(m)

        return len(good_matches), good_matches

    def filter_geometric_inliers(self, kpts1, kpts2, matches):

        if len(matches) < 4:
            return len(matches), matches
        
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            _, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thresh
            )
            if mask is None:
                return len(matches), matches
            inlier_mask = mask.ravel() == 1
            inlier_matches = [matches[idx] for idx, is_in in enumerate(inlier_mask) if is_in]
            return len(inlier_matches), inlier_matches
        except Exception:
            return len(matches), matches

    def get_self_similarity(self, img_input):

        if isinstance(img_input, (str, Path)):
            path_str = str(img_input)
            if path_str in self._cache_self_sim:
                return self._cache_self_sim[path_str]
        else:
            path_str = None

        feats = self.extract_features(img_input, rotate_180=False)
        _, descs_s, _ = feats['sift']
        _, descs_b, _ = feats['model_b']

        s_ii_sift, _ = self.match_descriptors(descs_s, descs_s, model_type="sift")
        s_ii_b, _ = self.match_descriptors(descs_b, descs_b, model_type="model_b")

        s_ii_sift = max(1, s_ii_sift)
        s_ii_b = max(1, s_ii_b)

        res = {'sift': s_ii_sift, 'model_b': s_ii_b, 'akaze': s_ii_b}
        if path_str is not None:
            self._cache_self_sim[path_str] = res
        return res

    def compute_pair_similarity(self, img1, img2):

        f1_0 = self.extract_features(img1, rotate_180=False)
        f1_180 = self.extract_features(img1, rotate_180=True)
        f2_0 = self.extract_features(img2, rotate_180=False)

        self_1 = self.get_self_similarity(img1)
        self_2 = self.get_self_similarity(img2)

        # SIFT 0° vs 0° & 180° vs 0°
        _, matches_s0 = self.match_descriptors(f1_0['sift'][1], f2_0['sift'][1], model_type="sift")
        in_s0, in_matches_s0 = self.filter_geometric_inliers(f1_0['sift'][0], f2_0['sift'][0], matches_s0)

        _, matches_s180 = self.match_descriptors(f1_180['sift'][1], f2_0['sift'][1], model_type="sift")
        in_s180, in_matches_s180 = self.filter_geometric_inliers(f1_180['sift'][0], f2_0['sift'][0], matches_s180)

        if in_s180 > in_s0:
            best_sift_inliers = in_s180
            best_sift_kpts1 = f1_180['sift'][0]
            sift_match_list = in_matches_s180
            sift_rot180 = True
        else:
            best_sift_inliers = in_s0
            best_sift_kpts1 = f1_0['sift'][0]
            sift_match_list = in_matches_s0
            sift_rot180 = False

        # Model B (AKAZE/ORB) 0° vs 0° & 180° vs 0°
        _, matches_b0 = self.match_descriptors(f1_0['model_b'][1], f2_0['model_b'][1], model_type="model_b")
        in_b0, in_matches_b0 = self.filter_geometric_inliers(f1_0['model_b'][0], f2_0['model_b'][0], matches_b0)

        _, matches_b180 = self.match_descriptors(f1_180['model_b'][1], f2_0['model_b'][1], model_type="model_b")
        in_b180, in_matches_b180 = self.filter_geometric_inliers(f1_180['model_b'][0], f2_0['model_b'][0], matches_b180)

        if in_b180 > in_b0:
            best_b_inliers = in_b180
            best_b_kpts1 = f1_180['model_b'][0]
            b_match_list = in_matches_b180
            b_rot180 = True
        else:
            best_b_inliers = in_b0
            best_b_kpts1 = f1_0['model_b'][0]
            b_match_list = in_matches_b0
            b_rot180 = False

        # Compute Directed and Symmetrized Ratio Scores
        r_ij_sift_dir = best_sift_inliers / self_1['sift']
        r_ji_sift_dir = best_sift_inliers / self_2['sift']
        r_sift_sym = min(r_ij_sift_dir, r_ji_sift_dir)

        r_ij_b_dir = best_b_inliers / self_1['model_b']
        r_ji_b_dir = best_b_inliers / self_2['model_b']
        r_b_sym = min(r_ij_b_dir, r_ji_b_dir)

        return {
            'sift_matches': best_sift_inliers,
            'sift_ratio': r_sift_sym,
            'sift_ratio_pct': r_sift_sym * 100.0,
            'sift_rot180': sift_rot180,
            'sift_match_list': sift_match_list,
            'sift_kpts1': best_sift_kpts1,
            'sift_kpts2': f2_0['sift'][0],
            
            'akaze_matches': best_b_inliers,
            'akaze_ratio': r_b_sym,
            'akaze_ratio_pct': r_b_sym * 100.0,
            'akaze_rot180': b_rot180,
            'akaze_match_list': b_match_list,
            'akaze_kpts1': best_b_kpts1,
            'akaze_kpts2': f2_0['model_b'][0],
            
            'self_sim_1': self_1,
            'self_sim_2': self_2
        }

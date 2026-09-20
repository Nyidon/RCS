
# preprocessing.py (Throat + Belly Pipeline)


import os
import cv2
import numpy as np
from pathlib import Path


def rotate_image(image, angle):
    """Rotates an image around its center without cropping corners."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    return cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compute_symmetry_iou(mask_patch):
    """Computes vertical Jaccard IoU between flipped left and right halves."""
    h, w = mask_patch.shape[:2]
    mid = w // 2
    left = mask_patch[:, :mid]
    right = mask_patch[:, mid:]

    min_w = min(left.shape[1], right.shape[1])
    if min_w == 0:
        return 0.0

    left_half = left[:, :min_w]
    right_half = right[:, :min_w]
    left_flipped = cv2.flip(left_half, 1)

    inter = np.logical_and(left_flipped > 0, right_half > 0).sum()
    union = np.logical_or(left_flipped > 0, right_half > 0).sum()
    return inter / union if union > 0 else 0.0


def find_optimal_symmetry_angle(img):
    """Multi-scale coarse-to-fine sweep from 0° to 180° to find the vertical midline axis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255

    # Coarse sweep: 0 to 180 in 2 deg steps
    best_angle = 0.0
    best_iou = -1.0
    for angle in np.arange(0, 180, 2.0):
        rot_mask = rotate_image(mask, angle)
        iou = compute_symmetry_iou(rot_mask)
        if iou > best_iou:
            best_iou = iou
            best_angle = angle

    # Fine sweep: +/- 3 deg in 0.5 deg steps
    for angle in np.arange(best_angle - 3.0, best_angle + 3.5, 0.5):
        rot_mask = rotate_image(mask, angle)
        iou = compute_symmetry_iou(rot_mask)
        if iou > best_iou:
            best_iou = iou
            best_angle = angle

    return best_angle


def score_ventral_orientation(img):
    """
    Evaluates whether the chin is North (Score > 0) or South (Score < 0).
    For full ventral patterns:
    - Chin (top) is narrower with corner voids.
    - Abdomen (bottom) is wider with greater mass.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    h, w = mask.shape[:2]

    # Top third vs Bottom third width
    top_third = mask[:h//3, :]
    bot_third = mask[2*h//3:, :]

    top_mass = (top_third > 0).sum()
    bot_mass = (bot_third > 0).sum()

    # Corner void difference (Top bounding corners should have more void created by chin arch)
    cw = max(1, w // 4)
    ch = max(1, h // 4)
    tl_void = (mask[:ch, :cw] == 0).sum()
    tr_void = (mask[:ch, w-cw:] == 0).sum()
    bl_void = (mask[h-ch:, :cw] == 0).sum()
    br_void = (mask[h-ch:, w-cw:] == 0).sum()

    corner_score = (tl_void + tr_void) - (bl_void + br_void)
    mass_score = bot_mass - top_mass  # Normal orientation has more mass at the bottom belly

    return (corner_score * 2.0) + (mass_score * 1.5)


def letterbox_pad_to_square(img, target_size=256):
    """
    Resizes image while preserving aspect ratio and pads with pure black borders to target_size x target_size.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    nw = int(w * scale)
    nh = int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Place in center of square black canvas
    square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - nh) // 2
    x_off = (target_size - nw) // 2
    square[y_off:y_off+nh, x_off:x_off+nw] = resized
    return square


def enhance_photometrics_rgb(img):
    """
    LAB Bilateral Filtering + CLAHE + Scalpel Detail Sharpening with strict black background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg_mask = (gray > 10).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Bilateral filter on L channel to suppress specular camera flash noise
    L_smooth = cv2.bilateralFilter(L, d=5, sigmaColor=35, sigmaSpace=35)

    # CLAHE on L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L_smooth)

    # High-pass Scalpel Unsharp Masking
    L_blur = cv2.GaussianBlur(L_clahe, (5, 5), 1.0)
    L_detail = cv2.subtract(L_clahe, L_blur)
    L_sharp = cv2.addWeighted(L_clahe, 1.0, L_detail, 1.8, 0)

    enhanced_lab = cv2.merge([L_sharp, A, B])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Strictly zero out background pixels
    enhanced_rgb[fg_mask == 0] = [0, 0, 0]
    return enhanced_rgb


def process_throat_belly_crops(input_base, output_base):
    print("\n" + "=" * 70)
    print("      PHASE 2: STANDARDIZING THROAT + BELLY CROPS (256x256 RGB)       ")
    print("=" * 70)

    splits = ['train', 'val']
    for split in splits:
        in_folder = os.path.join(input_base, split)
        out_folder = os.path.join(output_base, split)
        os.makedirs(out_folder, exist_ok=True)

        if not os.path.exists(in_folder):
            continue

        files = [f for f in os.listdir(in_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Standardizing [{split.upper()}]: {len(files)} crops...")

        count = 0
        for f in files:
            img_path = os.path.join(in_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Symmetry Alignment
            best_angle = find_optimal_symmetry_angle(img)
            aligned = rotate_image(img, best_angle)

            # 2. Anatomical Orientation Check (Strict Chin North)
            orientation_score = score_ventral_orientation(aligned)
            if orientation_score < 0:
                aligned = rotate_image(aligned, 180.0)

            # 3. Crop tightly to foreground bounding box
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            cnts, _ = cv2.findContours((gray > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                bx, by, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                aligned = aligned[by:by+bh, bx:bx+bw]

            # 4. Aspect-Preserving Letterbox Pad to 256x256
            padded = letterbox_pad_to_square(aligned, target_size=256)

            # 5. Photometric CLAHE & Sharpening
            final_crop = enhance_photometrics_rgb(padded)

            save_path = os.path.join(out_folder, f)
            cv2.imwrite(save_path, final_crop)
            count += 1

        print(f"✅ Finished [{split.upper()}]: Saved {count} standardized 256x256 crops.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    input_data = os.path.join(project_root, 'data', 'SAM2_Data')
    output_data = os.path.join(project_root, 'data', 'preprocessed')

    process_throat_belly_crops(input_data, output_data)

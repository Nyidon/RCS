import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

TARGET_SIZE = (256, 256)


def compute_symmetry_iou(mask_patch):

    # Computes normalized bilateral IoU (Jaccard Index) across the vertical centerline.
    h, w = mask_patch.shape[:2]
    mid = w // 2
    left_half = mask_patch[:, :mid]
    right_half = mask_patch[:, mid:]
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w <= 0 or h <= 0:
        return 0.0
    left_flipped = cv2.flip(left_half[:, :min_w], 1)
    right_trimmed = right_half[:, :min_w]
    overlap = cv2.countNonZero(cv2.bitwise_and(left_flipped, right_trimmed))
    union = cv2.countNonZero(cv2.bitwise_or(left_flipped, right_trimmed))
    return overlap / union if union > 0 else 0.0


def score_chin_north(crop_mask):

    h, w = crop_mask.shape[:2]
    if h < 5 or w < 5:
        return 0.0

    # 1. Longitudinal Width Gradient
    h_25 = max(1, int(h * 0.25))
    w_top = np.mean([np.count_nonzero(crop_mask[r, :]) for r in range(h_25)])
    w_bot = np.mean([np.count_nonzero(crop_mask[h - 1 - r, :]) for r in range(h_25)])
    s_width = (w_bot - w_top) / (w_bot + w_top + 1e-5)

    # 2. Corner Void Space Difference
    w_corner = max(1, w // 3)
    top_corner_void = (h_25 * w_corner * 2) - (
        np.count_nonzero(crop_mask[:h_25, :w_corner]) + np.count_nonzero(crop_mask[:h_25, -w_corner:])
    )
    bot_corner_void = (h_25 * w_corner * 2) - (
        np.count_nonzero(crop_mask[-h_25:, :w_corner]) + np.count_nonzero(crop_mask[-h_25:, -w_corner:])
    )
    s_corner = (top_corner_void - bot_corner_void) / (top_corner_void + bot_corner_void + 1e-5)

    # 3. Area Mass Distribution
    top_area = np.count_nonzero(crop_mask[:h // 2, :])
    bot_area = np.count_nonzero(crop_mask[h // 2:, :])
    s_area = (bot_area - top_area) / (top_area + bot_area + 1e-5)

    # 4. Vertical Moments (Center of Mass)
    M = cv2.moments(crop_mask)
    s_moment = ((M['m01'] / M['m00']) - (h / 2.0)) / (h / 2.0) if M['m00'] > 0 else 0.0

    return 3.5 * s_width + 2.5 * s_corner + 2.0 * s_area + 1.5 * s_moment


def align_chin_to_yaxis_rgb(img):

    if img is None:
        return None, None, 0, False

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    _, mask = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask, 0, False

    largest_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, cv2.FILLED)

    h, w = img.shape[:2]
    pad = max(h, w)
    padded_mask = cv2.copyMakeBorder(clean_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cy, cx = padded_mask.shape[0] // 2, padded_mask.shape[1] // 2

    # 1. Full 180° Projective Symmetry Sweep (Coarse in 2° steps)
    best_angle = 0
    best_obj = -1e9

    for angle in range(0, 180, 2):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot_mask = cv2.warpAffine(padded_mask, M, (padded_mask.shape[1], padded_mask.shape[0]))
        x, y, bw, bh = cv2.boundingRect(rot_mask)
        if bw <= 1 or bh <= 1:
            continue
        crop_m = rot_mask[y:y + bh, x:x + bw]
        iou = compute_symmetry_iou(crop_m)
        if iou > best_obj:
            best_obj = iou
            best_angle = angle

    # 2. Fine-Tuning Sweep (±3° in 0.5° steps)
    fine_angle = best_angle
    best_fine_obj = -1e9
    for angle in np.arange(best_angle - 3.0, best_angle + 3.5, 0.5):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot_mask = cv2.warpAffine(padded_mask, M, (padded_mask.shape[1], padded_mask.shape[0]))
        x, y, bw, bh = cv2.boundingRect(rot_mask)
        if bw <= 1 or bh <= 1:
            continue
        crop_m = rot_mask[y:y + bh, x:x + bw]
        iou = compute_symmetry_iou(crop_m)
        if iou > best_fine_obj:
            best_fine_obj = iou
            fine_angle = angle

    M_final = cv2.getRotationMatrix2D((cx, cy), fine_angle, 1.0)
    aligned_img = cv2.warpAffine(
        padded_img, M_final, (padded_img.shape[1], padded_img.shape[0]),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    aligned_mask = cv2.warpAffine(
        padded_mask, M_final, (padded_mask.shape[1], padded_mask.shape[0]),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    x, y, bw, bh = cv2.boundingRect(aligned_mask)
    if bw <= 0 or bh <= 0:
        return img, clean_mask, fine_angle, False

    crop_mask = aligned_mask[y:y + bh, x:x + bw]
    crop_img = aligned_img[y:y + bh, x:x + bw]

    # 3. Anatomical Upright Check (Chin to North)
    score = score_chin_north(crop_mask)
    flipped = False
    if score < 0:
        crop_img = cv2.rotate(crop_img, cv2.ROTATE_180)
        crop_mask = cv2.rotate(crop_mask, cv2.ROTATE_180)
        flipped = True

    return crop_img, crop_mask, fine_angle, flipped


def enhance_and_sharpen_rgb(img_bgr, mask, clip_limit=3.0, tile_grid_size=(8, 8)):

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1. Edge-preserving bilateral denoising on L
    l_denoised = cv2.bilateralFilter(l, d=5, sigmaColor=35, sigmaSpace=35)

    # 2. CLAHE local contrast enhancement on L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_denoised)

    # 3. Multi-scale Unsharp Masking detail boost on L
    l_blur = cv2.GaussianBlur(l_clahe, (0, 0), sigmaX=1.5)
    l_detail = l_clahe.astype(np.float32) - l_blur.astype(np.float32)
    l_sharp = np.clip(l_clahe.astype(np.float32) + 1.8 * l_detail, 0, 255).astype(np.uint8)

    # Re-enforce mask on L
    l_sharp[mask == 0] = 0

    # 4. Merge back with chromatic channels (zero color fringing)
    lab_enhanced = cv2.merge([l_sharp, a, b])
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    bgr_enhanced[mask == 0] = 0

    return bgr_enhanced


def preprocess_throat_rgb(img, target_size=TARGET_SIZE, clip_limit=3.0):
    """
    Complete RGB Preprocessing Pipeline:
    1. Upright chin-to-North alignment (Symmetry sweep + Anatomical scorer)
    2. Lanczos-4 high-fidelity resizing to target size (256x256)
    3. LAB Bilateral Denoising + L-channel CLAHE + Unsharp Masking
    4. Pure black background enforcement
    """
    if img is None:
        return None

    aligned_img, aligned_mask, _, _ = align_chin_to_yaxis_rgb(img)

    resized_bgr = cv2.resize(aligned_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(aligned_mask, target_size, interpolation=cv2.INTER_NEAREST)

    final_rgb = enhance_and_sharpen_rgb(resized_bgr, resized_mask, clip_limit=clip_limit)
    return final_rgb


def process_dataset(input_dir, output_dir, target_size=TARGET_SIZE, clip_limit=3.0):
    """Processes all images in input_dir and saves data RGB images to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = sorted([
        f for f in os.listdir(input_path)
        if Path(f).suffix.lower() in supported_exts and not f.startswith('.')
    ])

    print(f"\n{'='*70}")
    print(f"🚀 Starting RGB Preprocessing Pipeline (Upright Chin-North + LAB USM)")
    print(f"📂 Input Directory:  {input_path}")
    print(f"📁 Output Directory: {output_path}")
    print(f"🖼️ Found {len(image_files)} images to process (Target: {target_size}, Color: RGB)")
    print(f"{'='*70}\n")

    processed_count = 0
    flipped_count = 0
    angles_adjusted = []

    for img_name in tqdm(image_files, desc="Preprocessing RGB images"):
        img_file_path = input_path / img_name
        img = cv2.imread(str(img_file_path))

        if img is None:
            print(f"⚠️ Warning: Could not read {img_name}. Skipping...")
            continue

        aligned_img, aligned_mask, angle, flipped = align_chin_to_yaxis_rgb(img)
        if flipped:
            flipped_count += 1
        angles_adjusted.append(angle)

        resized_bgr = cv2.resize(aligned_img, target_size, interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(aligned_mask, target_size, interpolation=cv2.INTER_NEAREST)

        final_rgb = enhance_and_sharpen_rgb(resized_bgr, resized_mask, clip_limit=clip_limit)

        save_file_path = output_path / img_name
        cv2.imwrite(str(save_file_path), final_rgb)
        processed_count += 1

    print(f"\n{'='*70}")
    print(f"✅ RGB Preprocessing Complete!")
    print(f"   • Total Images Processed: {processed_count} / {len(image_files)}")
    print(f"   • Chin Orientation Flipped to North: {flipped_count} ({flipped_count/max(processed_count,1)*100:.1f}%)")
    if angles_adjusted:
        print(f"   • Tilt Correction: Min={min(angles_adjusted)}°, Max={max(angles_adjusted)}°, Mean={np.mean(angles_adjusted):.1f}°")
    print(f"   • Saved to: {output_path.resolve()}")
    print(f"{'='*70}\n")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="RGB Preprocessing Pipeline for Toad Throat Recognition")
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Optional path to specific input images directory (default: processes train and val)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional path to specific output directory'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='Target image square resolution (default: 256)'
    )
    parser.add_argument(
        '--clahe_clip',
        type=float,
        default=3.0,
        help='CLAHE clip limit for LAB L-channel contrast enhancement (default: 3.0)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    if args.input_dir and args.output_dir:
        process_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=(args.size, args.size),
            clip_limit=args.clahe_clip
        )
    else:
        base_input = project_root / '02_Throat_Preprocessing' / 'data' / 'SAM2_Data'
        base_output = project_root / '03_2_Siamese_network' / 'data' / 'rgb'

        splits = ['train', 'val']
        for split in splits:
            in_folder = base_input / split
            out_folder = base_output / split

            if in_folder.exists():
                print(f"\n--- Processing RGB [{split.upper()}] Dataset ---")
                process_dataset(
                    input_dir=in_folder,
                    output_dir=out_folder,
                    target_size=(args.size, args.size),
                    clip_limit=args.clahe_clip
                )
            else:
                print(f"⚠️ Input folder does not exist: {in_folder}")

"""
evaluate_sam2.py
================
Evaluates SAM 2 zero-shot throat segmentation quality by measuring exact pixel-level
region of overlap against human manual polygon annotations (LabelMe JSON).

"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO, SAM

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent


def parse_strict_polygon(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data.get('imageHeight')
    w = data.get('imageWidth')
    img_name = data.get('imagePath', '')

    shapes = data.get('shapes', [])
    polygon_shape = None
    for s in shapes:
        if s.get('shape_type') == 'polygon':
            polygon_shape = s
            break

    if polygon_shape is None:
        return None, h, w, img_name

    raw_pts = polygon_shape.get('points', [])
    if len(raw_pts) < 3:
        return None, h, w, img_name

    pts = np.array(raw_pts, dtype=np.int32)
    return pts, h, w, img_name


def chaikin_smooth(points, iterations=3, closed=True):
    pts = np.array(points, dtype=np.float32).reshape(-1, 2)
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n] if closed else pts[min(i + 1, n - 1)]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        pts = np.array(new_pts, dtype=np.float32)
    return pts.astype(np.int32).reshape(-1, 1, 2)


def create_smooth_triangular_mask(raw_contour, img_shape, inset_ratio=0.025, approx_eps=0.015, chaikin_iters=3):
    raw_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(raw_mask, [raw_contour], -1, 255, cv2.FILLED)

    _, _, bw, bh = cv2.boundingRect(raw_contour)
    min_dim = min(bw, bh)
    inset_px = max(1, int(inset_ratio * min_dim))

    if inset_px > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset_px * 2 + 1, inset_px * 2 + 1))
        eroded_mask = cv2.erode(raw_mask, erode_kernel)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea) if contours else raw_contour
    else:
        main_contour = raw_contour

    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, approx_eps * peri, True)
    hull = cv2.convexHull(approx)

    hull_peri = cv2.arcLength(hull, True)
    simplified_hull = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
    if len(simplified_hull) < 4:
        simplified_hull = hull

    smooth_pts = chaikin_smooth(simplified_hull, iterations=chaikin_iters, closed=True)

    mask_canvas = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_canvas, [smooth_pts], -1, 255, cv2.FILLED)

    round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, round_kernel)
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_CLOSE, round_kernel)
    mask_canvas = cv2.GaussianBlur(mask_canvas, (7, 7), 0)

    return (mask_canvas > 127).astype(np.uint8) * 255


def compute_polygon_overlap_metrics(manual_mask, auto_mask):

    m_bool = manual_mask > 0
    a_bool = auto_mask > 0

    inter_px = int(np.logical_and(m_bool, a_bool).sum())
    union_px = int(np.logical_or(m_bool, a_bool).sum())
    m_area = int(m_bool.sum())
    a_area = int(a_bool.sum())

    mask_iou = inter_px / union_px if union_px > 0 else 0.0
    dice = (2.0 * inter_px) / (m_area + a_area) if (m_area + a_area) > 0 else 0.0
    recall = inter_px / m_area if m_area > 0 else 0.0
    precision = inter_px / a_area if a_area > 0 else 0.0

    # SAM 2 Mask Solidity
    contours, _ = cv2.findContours(auto_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(main_cnt)
        hull = cv2.convexHull(main_cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cnt_area / hull_area if hull_area > 0 else 0.0
    else:
        solidity = 0.0

    return {
        "mask_iou": float(mask_iou),
        "dice": float(dice),
        "recall": float(recall),
        "precision": float(precision),
        "solidity": float(solidity),
        "manual_polygon_area_px": m_area,
        "sam2_mask_area_px": a_area,
        "overlap_area_px": inter_px
    }


def render_4panel_visualization(img, manual_mask, auto_mask, file_stem, metrics, output_dir):
    """
    Renders high-contrast 4-panel comparison montage:
      1. Original Input Image
      2. Human Ground Truth Manual Polygon (Green)
      3. Automated YOLOv8 + SAM 2 Prediction (Blue/Cyan)
      4. Overlap Diagnostic Composite (Yellow: Match, Red: Spill, Green: Missed)
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = img.shape[:2]

    # Panel 1: Original Image with Header
    panel_orig = img.copy()
    cv2.rectangle(panel_orig, (0, 0), (w, 36), (20, 24, 33), -1)
    cv2.putText(panel_orig, "1. Original Survey Photograph", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Panel 2: Manual Polygon overlay (Emerald Green)
    panel_m = img.copy()
    overlay_m = np.zeros_like(img)
    overlay_m[manual_mask > 0] = [16, 185, 129]  # Emerald Green
    panel_m = cv2.addWeighted(panel_m, 0.65, overlay_m, 0.35, 0)
    cnts_m, _ = cv2.findContours(manual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_m, cnts_m, -1, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.rectangle(panel_m, (0, 0), (w, 36), (20, 24, 33), -1)
    cv2.putText(panel_m, "2. Ground Truth Human Polygon", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (52, 211, 153), 2, cv2.LINE_AA)

    # Panel 3: SAM 2 Auto overlay (Royal Blue / Cyan)
    panel_a = img.copy()
    overlay_a = np.zeros_like(img)
    overlay_a[auto_mask > 0] = [235, 99, 37]  # Royal Blue (BGR)
    panel_a = cv2.addWeighted(panel_a, 0.65, overlay_a, 0.35, 0)
    cnts_a, _ = cv2.findContours(auto_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel_a, cnts_a, -1, (235, 99, 37), 2, cv2.LINE_AA)
    cv2.rectangle(panel_a, (0, 0), (w, 36), (20, 24, 33), -1)
    cv2.putText(panel_a, "3. Auto YOLOv8 + SAM 2 Extraction", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (244, 114, 182), 2, cv2.LINE_AA)

    # Panel 4: Color-coded Overlap Composite
    panel_ov = (img.copy() * 0.35).astype(np.uint8)
    manual_only = np.logical_and(manual_mask > 0, auto_mask == 0)
    auto_only = np.logical_and(auto_mask > 0, manual_mask == 0)
    overlap = np.logical_and(manual_mask > 0, auto_mask > 0)

    panel_ov[manual_only] = [0, 200, 0]      # Green = Undersegmented / Missed
    panel_ov[auto_only] = [0, 0, 230]        # Red = Oversegmented / Spill
    panel_ov[overlap] = [0, 230, 255]        # Yellow = True Positive Overlap

    cv2.rectangle(panel_ov, (0, 0), (w, 36), (20, 24, 33), -1)
    title = f"4. Overlap (Yellow) | IoU: {metrics['mask_iou']*100:.1f}% | Dice: {metrics['dice']*100:.1f}% | Prec: {metrics['precision']*100:.1f}%"
    cv2.putText(panel_ov, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # Combine 2x2 Montage
    top = np.hstack([panel_orig, panel_m])
    bot = np.hstack([panel_a, panel_ov])
    composite = np.vstack([top, bot])

    # Downscale for crisp storage if large
    max_dim = 1600
    gh, gw = composite.shape[:2]
    if max(gh, gw) > max_dim:
        scale = max_dim / max(gh, gw)
        composite = cv2.resize(composite, (int(gw * scale), int(gh * scale)), interpolation=cv2.INTER_AREA)

    save_path = os.path.join(output_dir, f"{file_stem}_polygon_overlap.jpg")
    cv2.imwrite(save_path, composite)
    return save_path


def run_polygon_evaluation(data_dir, weights_path, sam_weights_path, output_dir, max_viz=30):
    print("\n" + "=" * 75)
    print("      STRICT MANUAL POLYGON VS. SAM 2 CROPPING EVALUATION BENCHMARK       ")
    print("=" * 75)
    print(f"📁 Target Folder:      {data_dir}")
    print(f"🧠 YOLO Model:         {weights_path}")
    print(f"🎯 SAM 2 Model:        {sam_weights_path}")
    print("=" * 75 + "\n")

    yolo_model = YOLO(weights_path)
    sam_model = SAM(sam_weights_path)

    all_records = []
    viz_dir = os.path.join(output_dir, "visualizations") if output_dir else None
    viz_count = 0

    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return None

    json_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.json')])
    print(f"🔍 Found {len(json_files)} JSON annotation files in [{data_dir}].")

    polygon_count = 0
    for jf in json_files:
        json_path = os.path.join(data_dir, jf)
        stem = os.path.splitext(jf)[0]

        pts, h_anno, w_anno, img_name = parse_strict_polygon(json_path)
        if pts is None:
            continue  # Skip non-polygon shapes

        polygon_count += 1

        # Find matching image file
        img_path = None
        for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
            candidate = os.path.join(data_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None and img_name:
            candidate = os.path.join(data_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate

        if img_path is None:
            print(f"⚠️ Image for {stem} not found, skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 1. Rasterize Human Manual Polygon Mask Canvas
        manual_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(manual_mask, [pts], 255)

        # 2. Automated YOLO Localization
        res = yolo_model.predict(source=img_path, device='mps', conf=0.25, max_det=1, imgsz=640, verbose=False)
        if not res or len(res[0].boxes) == 0:
            all_records.append({
                "image": stem,
                "detected": False,
                "mask_iou": 0.0,
                "dice": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "solidity": 0.0,
                "manual_polygon_area_px": int((manual_mask > 0).sum()),
                "sam2_mask_area_px": 0,
                "overlap_area_px": 0
            })
            continue

        box = list(map(int, res[0].boxes.xyxy[0].tolist()))
        x1, y1, x2, y2 = box

        # 3. SAM 2 Throat Segmentation
        sam_res = sam_model.predict(source=img_path, bboxes=[x1, y1, x2, y2], device='mps', save=False, verbose=False)
        if sam_res[0].masks is None or len(sam_res[0].masks.xy) == 0:
            auto_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            raw_contour = sam_res[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
            auto_mask = create_smooth_triangular_mask(raw_contour, img.shape)

        # 4. Compute Pixel Overlap Metrics
        metrics = compute_polygon_overlap_metrics(manual_mask, auto_mask)
        metrics["image"] = stem
        metrics["detected"] = True
        all_records.append(metrics)

        # 5. Save visual comparison overlays
        if viz_dir and viz_count < max_viz:
            render_4panel_visualization(img, manual_mask, auto_mask, stem, metrics, viz_dir)
            viz_count += 1

    df = pd.DataFrame(all_records)
    if df.empty:
        print("\n❌ No polygon annotations (shape_type == 'polygon') found in the targeted directory.")
        return None

    # Print Formatted Results Table
    print("\n" + "=" * 75)
    print("               POLYGON OVERLAP BENCHMARK SUMMARY RESULTS                  ")
    print("=" * 75)
    print(f"Total Polygons Evaluated:        {len(df):,}")
    print(f"YOLO Throat Detection Rate:      {df['detected'].mean()*100:.2f}% ({df['detected'].sum()}/{len(df)})")
    print("-" * 75)
    print(f"Mean Mask IoU (Jaccard Index):   {df['mask_iou'].mean()*100:.2f}%  ± {df['mask_iou'].std()*100:.2f}%")
    print(f"Median Mask IoU:                 {df['mask_iou'].median()*100:.2f}%")
    print(f"Mean Dice Coefficient (F1):      {df['dice'].mean()*100:.2f}%  ± {df['dice'].std()*100:.2f}%")
    print(f"Mean Coverage / Recall:          {df['recall'].mean()*100:.2f}%  (Manual throat captured by SAM 2)")
    print(f"Mean Precision:                  {df['precision'].mean()*100:.2f}%  (Mask border cleanliness)")
    print(f"Mean SAM 2 Mask Solidity:        {df['solidity'].mean():.4f}  (Shape compactness)")
    print("=" * 75)

    # Save CSV Report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "sam2_polygon_overlap_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Detailed CSV report saved to: {csv_path}")
        if viz_dir:
            print(f"🖼️ Visual comparison maps saved to: {viz_dir}")

        # Generate 2-panel publication plot (Panels A and B)
        plot_path_eval = os.path.join(output_dir, "sam2_segmentation_performance.png")
        plot_path_plots = str(project_root / "results" / "plots" / "sam2_segmentation_performance.png")
        plot_sam2_segmentation_performance(df, plot_path_eval)
        plot_sam2_segmentation_performance(df, plot_path_plots)
        print(f"📈 Saved 2-panel performance plots to:\n   - {plot_path_eval}\n   - {plot_path_plots}")

    return df


def plot_sam2_segmentation_performance(df, output_path):
    """
    Renders 2-panel publication figure:
      Panel A: Mask IoU & Dice Similarity Distribution
      Panel B: Segmentation Precision vs. Coverage Recall
    """
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.suptitle("Phase 1: Zero-Shot SAM 2 Throat Segmentation vs. Manual Ground Truth", fontsize=15, fontweight="bold", y=0.98)

    # Panel A: Mask IoU & Dice Similarity Distribution
    iou_scores = df["mask_iou"] * 100
    dice_scores = df["dice"] * 100

    bins = np.linspace(45, 95, 21)
    ax1.hist(iou_scores, bins=bins, color="#0284c7", alpha=0.8, edgecolor="black", linewidth=0.8, label=f"Mask IoU (μ={iou_scores.mean():.1f}%)")
    ax1.hist(dice_scores, bins=bins, color="#10b981", alpha=0.7, edgecolor="black", linewidth=0.8, label=f"Dice F1 (μ={dice_scores.mean():.1f}%)")
    ax1.axvline(iou_scores.mean(), color="#0369a1", linestyle="--", linewidth=2.0)
    ax1.axvline(dice_scores.mean(), color="#047857", linestyle="--", linewidth=2.0)

    ax1.set_title("A. Mask Overlap Distributions (N=70)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Overlap Score (%)", fontsize=11, fontweight="600")
    ax1.set_ylabel("Image Count", fontsize=11, fontweight="600")
    ax1.set_xlim(45, 98)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Segmentation Precision vs. Coverage Recall
    rec = df["recall"] * 100
    prec = df["precision"] * 100

    ax2.scatter(rec, prec, color="#8b5cf6", edgecolor="#6d28d9", s=65, alpha=0.75, zorder=4)
    ax2.axhline(prec.mean(), color="#ef4444", linestyle="--", linewidth=1.8, label=f"Mean Precision: {prec.mean():.1f}%")
    ax2.axvline(rec.mean(), color="#3b82f6", linestyle="--", linewidth=1.8, label=f"Mean Recall: {rec.mean():.1f}%")

    ax2.set_title("B. Border Precision vs. Coverage Recall Trade-Off", fontsize=12.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Coverage Recall (%) [Manual Throat Captured]", fontsize=11, fontweight="600")
    ax2.set_ylabel("Border Precision (%) [Border Cleanliness]", fontsize=11, fontweight="600")
    ax2.set_ylim(75, 102)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Evaluate SAM 2 Zero-Shot Throat Segmentation Overlap against Manual Polygons")
    parser.add_argument("--data_dir", type=str, default=str(project_root / "01_Throat_detection_pipeline" / "data" / "LabelMe_Test"),
                        help="Path to LabelMe_Test folder containing images and LabelMe JSON polygon files")
    parser.add_argument("--weights", type=str, default=str(script_dir / "detect" / "train" / "weights" / "best.pt"),
                        help="Path to trained YOLOv8 weights (best.pt)")
    parser.add_argument("--sam_weights", type=str, default=str(project_root / "sam2_t.pt"),
                        help="Path to SAM 2 weights (sam2_t.pt)")
    parser.add_argument("--output_dir", type=str, default=str(project_root / "results" / "sam2_polygon_evaluation"),
                        help="Output directory for CSV reports and visual overlap maps")
    parser.add_argument("--max_viz", type=int, default=70,
                        help="Max visual 4-panel comparison images to generate (default: 70)")

    args = parser.parse_args()

    # Fallback check for weights
    if not os.path.exists(args.weights):
        alt_weights = script_dir / "runs" / "detect" / "train" / "weights" / "best.pt"
        if alt_weights.exists():
            args.weights = str(alt_weights)

    if not os.path.exists(args.sam_weights):
        alt_sam = script_dir / "sam2_t.pt"
        if alt_sam.exists():
            args.sam_weights = str(alt_sam)

    run_polygon_evaluation(
        data_dir=args.data_dir,
        weights_path=args.weights,
        sam_weights_path=args.sam_weights,
        output_dir=args.output_dir,
        max_viz=args.max_viz
    )

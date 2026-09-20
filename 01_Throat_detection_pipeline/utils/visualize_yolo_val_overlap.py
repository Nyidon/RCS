import os
import random
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Format: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def parse_yolo_txt(txt_path, img_w, img_h):
    """
    Parse YOLO format text file: class_id x_center y_center width height (normalized)
    Returns list of [x1, y1, x2, y2] in absolute pixel coordinates.
    """
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, xc, yc, w, h = map(float, parts[:5])
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append([x1, y1, x2, y2])
    return boxes

def get_zoom_crop_coords(gt_boxes, pred_boxes, img_w, img_h, margin_ratio=0.35):
    all_boxes = gt_boxes + pred_boxes
    if not all_boxes:
        return 0, 0, img_w, img_h
    min_x = min(b[0] for b in all_boxes)
    min_y = min(b[1] for b in all_boxes)
    max_x = max(b[2] for b in all_boxes)
    max_y = max(b[3] for b in all_boxes)

    bw = max_x - min_x
    bh = max_y - min_y
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    cx1 = max(0, min_x - mx)
    cy1 = max(0, min_y - my)
    cx2 = min(img_w, max_x + mx)
    cy2 = min(img_h, max_y + my)
    return cx1, cy1, cx2, cy2

def render_comparison_plot(img_rgb, gt_boxes, pred_boxes, pred_confs, base_name, iou, out_path):
    """
    Renders a publication-quality 2-panel figure using Matplotlib vector patches.
    """
    h, w = img_rgb.shape[:2]
    cx1, cy1, cx2, cy2 = get_zoom_crop_coords(gt_boxes, pred_boxes, w, h)
    zoom_crop = img_rgb[cy1:cy2, cx1:cx2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=200)

    # --- Panel 1: Full Field Image ---
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Full Field Survey Photo\n{base_name[:35]}", fontsize=11, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Draw boxes on full image
    for gt in gt_boxes:
        rect = patches.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                                 linewidth=2.5, edgecolor='#00E676', facecolor='none', label='Manual Ground Truth')
        axes[0].add_patch(rect)

    for i, (pred, conf) in enumerate(zip(pred_boxes, pred_confs)):
        rect = patches.Rectangle((pred[0], pred[1]), pred[2] - pred[0], pred[3] - pred[1],
                                 linewidth=2.5, edgecolor='#FF3D00', facecolor='none', linestyle='--', label=f'YOLOv8 Pred ({conf:.2f})')
        axes[0].add_patch(rect)

    axes[0].legend(loc='upper right', framealpha=0.9, facecolor='white', fontsize=10)

    # --- Panel 2: High-Resolution Zoomed Throat Region ---
    axes[1].imshow(zoom_crop)
    conf_str = f"{pred_confs[0]:.2f}" if pred_confs else "N/A"
    axes[1].set_title(f"Zoomed Throat Crop Overlay\nIoU: {iou:.3f} ({iou*100:.1f}%) | YOLO Confidence: {conf_str}",
                      fontsize=12, fontweight='bold',
                      color='#1B5E20' if iou >= 0.8 else ('#E65100' if iou >= 0.7 else '#B71C1C'), pad=10)
    axes[1].axis('off')

    # Draw boxes on cropped coordinate space
    for gt in gt_boxes:
        rx = gt[0] - cx1
        ry = gt[1] - cy1
        rw = gt[2] - gt[0]
        rh = gt[3] - gt[1]
        rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3.5, edgecolor='#00E676', facecolor='none', label='Manual Ground Truth')
        axes[1].add_patch(rect)

    for pred, conf in zip(pred_boxes, pred_confs):
        rx = pred[0] - cx1
        ry = pred[1] - cy1
        rw = pred[2] - pred[0]
        rh = pred[3] - pred[1]
        rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3.5, edgecolor='#FF3D00', facecolor='none', linestyle='--', label=f'YOLOv8 Detection ({conf:.2f})')
        axes[1].add_patch(rect)

    axes[1].legend(loc='upper right', framealpha=0.9, facecolor='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO Validation vs Ground Truth Bounding Box Overlaps")
    parser.add_argument("--weights", type=str, default="01_Throat_detection_pipeline/src/detect/train/weights/best.pt", help="Path to YOLO weights")
    parser.add_argument("--val-dir", type=str, default="01_Throat_detection_pipeline/data/val", help="Path to validation directory")
    parser.add_argument("--output-dir", type=str, default="results/yolo_val_overlap", help="Output directory for visual comparisons")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of random samples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO prediction")
    parser.add_argument("--device", type=str, default="mps", help="Device to run YOLO (mps/cpu/cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "individual"), exist_ok=True)

    print(f"Loading YOLO model from: {args.weights}")
    model = YOLO(args.weights)

    # Collect valid validation image paths (having matching .txt)
    all_files = os.listdir(args.val_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    valid_pairs = []

    for f in all_files:
        base, ext = os.path.splitext(f)
        if ext in image_exts:
            txt_file = os.path.join(args.val_dir, f"{base}.txt")
            if os.path.exists(txt_file):
                valid_pairs.append((os.path.join(args.val_dir, f), txt_file, base))

    print(f"Total valid annotated validation images found: {len(valid_pairs)}")
    if len(valid_pairs) == 0:
        print("Error: No validation images with annotations found!")
        return

    # Random sampling
    if args.seed is not None:
        random.seed(args.seed)
    
    sample_count = min(args.num_samples, len(valid_pairs))
    sampled_pairs = random.sample(valid_pairs, sample_count)

    print(f"\nVisualizing {sample_count} random samples with GT vs YOLO overlap...")
    
    results_summary = []
    grid_data = []

    for idx, (img_path, txt_path, base_name) in enumerate(sampled_pairs, 1):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Parse Ground Truth
        gt_boxes = parse_yolo_txt(txt_path, w, h)

        # 2. Run YOLO Prediction
        preds = model.predict(source=img_bgr, conf=args.conf, device=args.device, verbose=False)[0]
        pred_boxes = []
        pred_confs = []
        if len(preds.boxes) > 0:
            for box in preds.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                pred_boxes.append(xyxy)
                pred_confs.append(conf)

        # 3. Calculate IoU with primary GT
        iou = calculate_iou(gt_boxes[0], pred_boxes[0]) if (gt_boxes and pred_boxes) else 0.0
        top_conf = pred_confs[0] if pred_confs else 0.0

        # Save individual 2-panel comparison
        single_save_path = os.path.join(args.output_dir, "individual", f"sample_{idx:02d}_{base_name[:25]}.png")
        render_comparison_plot(img_rgb, gt_boxes, pred_boxes, pred_confs, base_name, iou, single_save_path)

        cx1, cy1, cx2, cy2 = get_zoom_crop_coords(gt_boxes, pred_boxes, w, h)
        zoom_crop = img_rgb[cy1:cy2, cx1:cx2]
        grid_data.append({
            "name": base_name,
            "crop": zoom_crop,
            "gt_rel": [[b[0]-cx1, b[1]-cy1, b[2]-b[0], b[3]-b[1]] for b in gt_boxes],
            "pred_rel": [[b[0]-cx1, b[1]-cy1, b[2]-b[0], b[3]-b[1]] for b in pred_boxes],
            "iou": iou,
            "conf": top_conf
        })

        results_summary.append({
            "index": idx,
            "filename": os.path.basename(img_path),
            "yolo_conf": top_conf,
            "iou": iou,
            "saved_to": single_save_path
        })
        print(f"[{idx:02d}/{sample_count:02d}] {base_name[:35]}: IoU = {iou:.3f}, YOLO Conf = {top_conf:.2f}")

    # Generate Combined 10-Image Montage Grid (2 rows x 5 columns)
    print("\nGenerating clean 10-image summary montage grid...")
    cols = 5
    rows = int(np.ceil(sample_count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(22, 5.2 * rows), dpi=250)
    fig.subplots_adjust(top=0.88, bottom=0.05, hspace=0.35, wspace=0.15)
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < len(grid_data):
            item = grid_data[i]
            axes[i].imshow(item["crop"])
            
            # Draw GT boxes (solid green)
            for gx, gy, gw, gh in item["gt_rel"]:
                rect = patches.Rectangle((gx, gy), gw, gh, linewidth=3, edgecolor='#00E676', facecolor='none')
                axes[i].add_patch(rect)
                
            # Draw Pred boxes (dashed orange)
            for px, py, pw, ph in item["pred_rel"]:
                rect = patches.Rectangle((px, py), pw, ph, linewidth=3, edgecolor='#FF3D00', facecolor='none', linestyle='--')
                axes[i].add_patch(rect)

            iou_val = item["iou"]
            conf_val = item["conf"]
            axes[i].set_title(f"Sample #{i+1}: IoU = {iou_val:.3f}\nYOLO Conf = {conf_val:.2f}",
                              fontsize=11, fontweight='bold',
                              color='#1B5E20' if iou_val >= 0.8 else ('#E65100' if iou_val >= 0.7 else '#B71C1C'), pad=8)
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    fig.suptitle("YOLOv8 Detection vs. Manual Ground Truth Bounding Box Overlap (10 Validation Samples)\n"
                 "Solid Green Line = Manual Ground Truth  |  Dashed Orange Line = YOLOv8 Prediction",
                 fontsize=15, fontweight='bold', y=0.96)
    montage_path = os.path.join(args.output_dir, "yolo_gt_vs_val_overlap_montage_10.png")
    plt.savefig(montage_path, bbox_inches='tight')
    plt.close(fig)

    # Print summary statistics
    ious = [r["iou"] for r in results_summary]
    confs = [r["yolo_conf"] for r in results_summary]
    print("\n" + "="*60)
    print("      YOLO VALIDATION VS GROUND TRUTH SUMMARY (10 SAMPLES)   ")
    print("="*60)
    print(f"Mean IoU Overlap:          {np.mean(ious):.4f} (Min: {np.min(ious):.4f}, Max: {np.max(ious):.4f})")
    print(f"Mean YOLO Confidence:      {np.mean(confs):.4f}")
    print(f"IoU >= 0.80 Count:         {sum(1 for x in ious if x >= 0.80)}/{len(ious)}")
    print(f"Combined Grid Saved to:    {montage_path}")
    print(f"Individual Plots Saved to: {os.path.join(args.output_dir, 'individual')}")
    print("="*60)

if __name__ == "__main__":
    main()

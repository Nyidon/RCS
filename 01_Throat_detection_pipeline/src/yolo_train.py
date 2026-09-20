import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM


def train_yolo_model(current_dir):
    print("\n--- PHASE 1: Training YOLO on Neck Bounding Boxes ---")
    yaml_path = os.path.join(current_dir, "dataset.yaml")
    model = YOLO('yolov8n.pt')

    # Start Training
    results = model.train(
        data=yaml_path,
        epochs=150,
        patience=25,
        imgsz=640,
        batch=16,
        workers=4,
        device='mps',
        amp=False,
        exist_ok=True
    )

    print("\n--- Evaluating on the Validation Set ---")
    metrics = model.val(device='mps')

    print("\n==================================================")
    print("           YOLOv8 VALIDATION METRICS              ")
    print("==================================================")
    print(f"Box Precision (P):       {metrics.box.mp:.4f}")
    print(f"Box Recall (R):          {metrics.box.mr:.4f}")
    print(f"Box mAP50:               {metrics.box.map50:.4f}")
    print(f"Box mAP50-95:            {metrics.box.map:.4f}")
    print("==================================================")

    save_dir = results.save_dir

    weights_path = os.path.join(save_dir, 'weights', 'best.pt')
    return weights_path

def chaikin_smooth(points, iterations=3, closed=True):

    # Chaikin's corner-cutting algorithm: Iteratively rounds off sharp polygon vertices into smooth, continuous curves.

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

    # 1. Base mask from raw SAM 2 contour
    raw_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(raw_mask, [raw_contour], -1, 255, cv2.FILLED)

    # 2. Mild adaptive inset to trim background noise while retaining the slim black anatomical border
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

    # 3. Simplify noisy boundary and extract convex hull envelope (bridges interior spots & yellow skin)
    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, approx_eps * peri, True)
    hull = cv2.convexHull(approx)

    # Decimate hull to primary triangular anchor vertices
    hull_peri = cv2.arcLength(hull, True)
    simplified_hull = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
    if len(simplified_hull) < 4:
        simplified_hull = hull

    # 4. Round off all sharp polygon corners into continuous curves
    smooth_pts = chaikin_smooth(simplified_hull, iterations=chaikin_iters, closed=True)

    # 5. Render smoothed mask canvas
    mask_canvas = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_canvas, [smooth_pts], -1, 255, cv2.FILLED)

    # 6. Elliptical opening and closing to guarantee smooth organic curvature
    round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, round_kernel)
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_CLOSE, round_kernel)

    # 7. Smooth anti-aliased edge transition
    mask_canvas = cv2.GaussianBlur(mask_canvas, (7, 7), 0)

    return mask_canvas


def generate_sam2_crops(current_dir, weights_path):
    print("\n--- PHASE 2: Starting YOLO + SAM 2 Extraction Pipeline ---")
    print(f"Loading YOLO Brain from: {weights_path}")

    yolo_model = YOLO(weights_path)
    sam_model = SAM('sam2_t.pt')
    project_root = os.path.dirname(current_dir)

    base_input = os.path.join(project_root, 'data')
    base_output = os.path.join(project_root, 'data', 'SAM2_Data')

    splits = ['train', 'val']
    for split in splits:
        input_folder = os.path.join(base_input, split)
        output_folder = os.path.join(base_output, split)
        os.makedirs(output_folder, exist_ok=True)
        for f in os.listdir(output_folder):
            file_path = os.path.join(output_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        image_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"\n Processing {split.upper()} Set: Found {len(image_files)} images in folder.")
        crop_counter = 1

        for filename in image_files:
            print(f"\n🔍 Analyzing: {filename}")
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print("   ❌ ERROR: Image file is corrupted or unreadable.")
                continue

            # Standard YOLO Bounding Box Detection
            res = yolo_model.predict(source=img_path, device='mps', conf=0.25, max_det=1, imgsz=640, verbose=False)

            if not res or len(res[0].boxes) == 0:
                print("   ❌ YOLO saw nothing.")
                continue

            box = list(map(int, res[0].boxes.xyxy[0].tolist()))
            x1, y1, x2, y2 = box
            print(f"   🎯 Bounding box located at: [{x1}, {y1}, {x2}, {y2}]. Handing to SAM 2...")

            # Check if SAM 2 finds the edges
            sam_results = sam_model.predict(source=img_path, bboxes=[x1, y1, x2, y2], device='mps', save=False, verbose=False)

            if sam_results[0].masks is None or len(sam_results[0].masks.xy) == 0:
                print("   ❌ SAM 2 failed to generate a shape inside the box.")
            else:
                raw_contour = sam_results[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)

                # Generate smoothed triangular throat mask with rounded vertices
                mask_canvas = create_smooth_triangular_mask(raw_contour, img.shape)

                # Pure Black Background Enforcement
                final_clean_toad = cv2.bitwise_and(img, img, mask=mask_canvas)

                # Crop the bounding box
                cropped_toad = final_clean_toad[y1:y2, x1:x2]
                original_name = os.path.splitext(os.path.basename(img_path))[0]
                save_name = f"{original_name}.jpg"

                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, cropped_toad)

                print(f"   ✅ SUCCESS: Saved {save_name}")
                crop_counter += 1

    print("\n✅ Total Extraction Complete!")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Check potential weights locations
    weights_candidates = [
        os.path.join(script_dir, 'detect', 'train', 'weights', 'best.pt'),
        os.path.join(script_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    ]
    stable_weights_path = next((p for p in weights_candidates if os.path.exists(p)), weights_candidates[0])

    run_training = True
    run_SAM2 = False

    if run_training:
        stable_weights_path = train_yolo_model(script_dir)
    if run_SAM2:
        generate_sam2_crops(script_dir, stable_weights_path)
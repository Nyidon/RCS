"""
convert_to_txt.py
=================
Converts LabelMe JSON annotations (rectangle or polygon) into normalized YOLO TXT format.
Target class: 0 (toad_throat_belly).
"""

import json
from pathlib import Path


def convert_labelme_to_yolo(folder_path, class_id=0):
    folder = Path(folder_path)
    if not folder.exists():
        return

    print(f"\n--- Scanning {folder.name} ---")
    json_files = list(folder.glob('*.json'))
    print(f"🔍 Found {len(json_files)} LabelMe annotations.")

    success_count = 0

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img_width = data['imageWidth']
        img_height = data['imageHeight']

        txt_path = json_path.with_suffix('.txt')

        with open(txt_path, 'w', encoding='utf-8') as out_file:
            for shape in data.get('shapes', []):
                pts = shape.get('points', [])
                if not pts:
                    continue

                if shape.get('shape_type') == 'rectangle':
                    x1, y1 = pts[0]
                    x2, y2 = pts[1]
                    xmin, xmax = min(x1, x2), max(x1, x2)
                    ymin, ymax = min(y1, y2), max(y1, y2)
                else:  # Polygon or other shape
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                box_width = xmax - xmin
                box_height = ymax - ymin
                x_center = xmin + (box_width / 2.0)
                y_center = ymin + (box_height / 2.0)

                # Normalize between 0.0 and 1.0
                norm_x = max(0.0, min(1.0, x_center / img_width))
                norm_y = max(0.0, min(1.0, y_center / img_height))
                norm_w = max(0.0, min(1.0, box_width / img_width))
                norm_h = max(0.0, min(1.0, box_height / img_height))

                out_file.write(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        success_count += 1

    print(f"✅ Converted {success_count} JSONs to TXT in {folder.name}!")


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    TRAIN_FOLDER = project_root / "data" / "train"
    VAL_FOLDER = project_root / "data" / "val"

    convert_labelme_to_yolo(TRAIN_FOLDER, class_id=0)
    convert_labelme_to_yolo(VAL_FOLDER, class_id=0)

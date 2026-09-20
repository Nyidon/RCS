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
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_width = data['imageWidth']
        img_height = data['imageHeight']

        # Create a text file with the same name, but .txt extension
        txt_path = json_path.with_suffix('.txt')

        with open(txt_path, 'w') as out_file:
            # Loop through every box drawn in that image
            for shape in data['shapes']:
                if shape['shape_type'] != 'rectangle':
                    continue

                    # LabelMe points: [[x1, y1], [x2, y2]]
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]

                # YOLO Math: Calculate center, width, and height
                box_width = abs(x2 - x1)
                box_height = abs(y2 - y1)
                x_center = min(x1, x2) + (box_width / 2)
                y_center = min(y1, y2) + (box_height / 2)

                # Normalize values between 0.0 and 1.0
                norm_x = x_center / img_width
                norm_y = y_center / img_height
                norm_w = box_width / img_width
                norm_h = box_height / img_height

                # Write to the YOLO format: "class_id x_center y_center width height"
                out_file.write(f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        success_count += 1

    print(f"✅ Converted {success_count} JSONs to TXT in {folder.name}!")


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Target your specific folders
    TRAIN_FOLDER = project_root / "data" / "train"
    VAL_FOLDER = project_root / "data" / "val"

    # Run the converter on both
    convert_labelme_to_yolo(TRAIN_FOLDER, class_id=0)
    convert_labelme_to_yolo(VAL_FOLDER, class_id=0)
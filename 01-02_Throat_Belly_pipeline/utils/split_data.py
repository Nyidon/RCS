"""
split_data.py
=============
Splits throat+belly raw images into stratified train and val partitions.
"""

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split


def split_throat_belly_dataset(raw_dir, output_base, train_size=300, random_state=42):
    print(f"🔍 Scanning images in: {raw_dir}")

    all_images = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if not file.startswith('.') and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))

    print(f"📊 Found {len(all_images)} total images.")

    if len(all_images) < train_size:
        print(f"❌ ERROR: Not enough images found ({len(all_images)} < {train_size})! Check your folder path.")
        return

    train_imgs, val_imgs = train_test_split(
        all_images,
        train_size=train_size,
        random_state=random_state
    )

    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Wipe old files to avoid leftover stale items
    for d in [train_dir, val_dir]:
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                os.remove(fp)

    print(f"Partitioning -> Train: {len(train_imgs)} images | Val: {len(val_imgs)} images")

    def copy_file_and_annotation(src_img, dest_folder):
        # Copy image
        fname = os.path.basename(src_img)
        shutil.copy2(src_img, os.path.join(dest_folder, fname))

        # Copy matching JSON if present
        json_src = os.path.splitext(src_img)[0] + '.json'
        if os.path.exists(json_src):
            shutil.copy2(json_src, os.path.join(dest_folder, os.path.basename(json_src)))

    for img_path in train_imgs:
        copy_file_and_annotation(img_path, train_dir)

    for img_path in val_imgs:
        copy_file_and_annotation(img_path, val_dir)

    print("✅ Throat+Belly dataset successfully split into train/ (300) and val/ (281)!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(description="Split Throat+Belly raw data into train (300) and val (remaining)")
    parser.add_argument("--raw_dir", type=str, default=os.path.join(project_root, 'data', 'raw'))
    parser.add_argument("--output_base", type=str, default=os.path.join(project_root, 'data'))
    parser.add_argument("--train_size", type=int, default=300)

    args = parser.parse_args()
    split_throat_belly_dataset(args.raw_dir, args.output_base, train_size=args.train_size)

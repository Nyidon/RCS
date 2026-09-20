import os
import shutil
from sklearn.model_selection import train_test_split


def split_toad_dataset(raw_images_folder, base_output_folder):
    print(f"🔍 Scanning nested folders inside: {raw_images_folder}")

    all_images = []

    for root, dirs, files in os.walk(raw_images_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                all_images.append(full_path)

    print(f"📊 Found {len(all_images)} total images across subfolders.")

    if len(all_images) < 300:
        print("❌ ERROR: Not enough images found! Check your folder path.")
        return

    train_images, val_images = train_test_split(
        all_images,
        train_size=300,
        random_state=20
    )

    train_dir = os.path.join(base_output_folder, 'train')
    val_dir = os.path.join(base_output_folder, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("\nCopying files")

    def copy_with_unique_name(src_path, dest_dir):
        # Generate relative path, e.g., 'Aulbachtal_2025/Tag_01/photo1.jpeg'
        rel_path = os.path.relpath(src_path, raw_images_folder)

        # Replace subfolder slashes with underscores, e.g., 'Aulbachtal_2025_Tag_01_photo1.jpeg'
        unique_filename = rel_path.replace(os.sep, '_')

        dest_path = os.path.join(dest_dir, unique_filename)
        shutil.copy(src_path, dest_path)

    for img_path in train_images:
        copy_with_unique_name(img_path, train_dir)

    for img_path in val_images:
        copy_with_unique_name(img_path, val_dir)

    print("✅ Dataset successfully split and saved with zero risk of overwrites!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_images_path = os.path.join(project_root, 'data', 'raw_data_2025')
    output_path = os.path.join(project_root, 'data')

    if os.path.exists(raw_images_path):
        split_toad_dataset(raw_images_path, output_path)
    else:
        print(f"❌ Could not find raw data path: {raw_images_path}")
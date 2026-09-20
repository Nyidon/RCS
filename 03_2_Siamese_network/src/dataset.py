import os
import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

TARGET_SIZE = (224, 224)


def get_default_transforms(is_train=True):
    """
    Carefully designed augmentations for ventral throat patterns:
    - 224x224 resolution preserves micro-spot (melanophore) sharpness.
    - Preserves anatomical orientation (no vertical flips).
    - Random rotation (+/- 8 degrees) and minor affine jitter to simulate neck movement.
    - Random Erasing / Cutout (p=0.35) forces holistic multi-spot reliance rather than single-spot overfitting.
    - Color jitter for lighting variance across cloudy/sunny field conditions.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomRotation(degrees=8, fill=0),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05), fill=0),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.15), value=0)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def load_clusters_from_toad_id(toad_id_dir=None, exclude_test_images=None):

    project_root = Path(__file__).resolve().parent.parent.parent
    if toad_id_dir is None:
        toad_id_dir = project_root / "03_2_Siamese_network" / "data" / "train" / "rgb_toad_id"
    else:
        toad_id_dir = Path(toad_id_dir)

    test_set = set(exclude_test_images) if exclude_test_images else set()
    clusters = {}
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    if not toad_id_dir.exists():
        manifest_path = toad_id_dir.parent / f"{toad_id_dir.name}_manifest.json"
        if not manifest_path.exists():
            manifest_path = toad_id_dir.parent / "rgb_toad_id_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for t_id, imgs in data.get("clusters", {}).items():
                paths = [str(toad_id_dir / t_id / img) for img in imgs if img not in test_set]
                if paths:
                    clusters[t_id] = paths
        return clusters

    subdirs = sorted([d for d in os.listdir(toad_id_dir) if (toad_id_dir / d).is_dir() and not d.startswith('.')])
    for d in subdirs:
        folder_path = toad_id_dir / d
        img_files = sorted([
            folder_path / f
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts) and not f.startswith('.') and f not in test_set
        ])
        if img_files:
            clusters[d] = [str(p) for p in img_files]

    return clusters


class ToadIdentityBatchDataset(Dataset):

    def __init__(self, clusters, transform=None):
        self.transform = transform if transform is not None else get_default_transforms(is_train=True)
        self.samples = []
        self.label_to_id = {}
        self.id_to_label = {}
        self.label_to_indices = {}

        # Build index
        for label_idx, (toad_id, img_paths) in enumerate(sorted(clusters.items())):
            self.label_to_id[label_idx] = toad_id
            self.id_to_label[toad_id] = label_idx
            self.label_to_indices[label_idx] = []

            for path in img_paths:
                sample_idx = len(self.samples)
                self.samples.append((path, label_idx))
                self.label_to_indices[label_idx].append(sample_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


class PKBatchSampler(Sampler):
    """
    P-K Batch Sampler for Online Hard-Negative Metric Learning:
    In every batch, samples P unique identity classes, and K images per class (BatchSize = P * K).
    """
    def __init__(self, label_to_indices, p=8, k=4, num_batches=100):
        self.label_to_indices = label_to_indices
        self.p = p
        self.k = k
        self.num_batches = num_batches
        self.multi_labels = [l for l, indices in label_to_indices.items() if len(indices) >= 2]
        self.all_labels = list(label_to_indices.keys())

        if len(self.multi_labels) < self.p // 2:
            self.multi_labels = self.all_labels

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            chosen_classes = set()
            if len(self.multi_labels) >= self.p:
                chosen_classes = set(random.sample(self.multi_labels, self.p))
            else:
                chosen_classes.update(self.multi_labels)
                remaining = self.p - len(chosen_classes)
                avail = [l for l in self.all_labels if l not in chosen_classes]
                if avail:
                    chosen_classes.update(random.sample(avail, min(remaining, len(avail))))

            for cls_label in chosen_classes:
                indices = self.label_to_indices[cls_label]
                if len(indices) >= self.k:
                    batch.extend(random.sample(indices, self.k))
                else:
                    batch.extend(random.choices(indices, k=self.k))

            yield batch


class ToadTripletDataset(Dataset):
    """
    Triplet Dataset for Deep Metric Learning:
    Yields (Anchor, Positive, Negative) triplets.
    """
    def __init__(self, clusters, transform=None, samples_per_epoch=2000):
        self.transform = transform if transform is not None else get_default_transforms(is_train=True)
        self.samples_per_epoch = samples_per_epoch
        self.clusters = clusters

        self.multi_sighting_ids = [t_id for t_id, imgs in clusters.items() if len(imgs) >= 2]
        self.all_toad_ids = list(clusters.keys())

        if not self.multi_sighting_ids:
            raise ValueError("No multi-sighting toads available to form positive pairs.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        anchor_id = random.choice(self.multi_sighting_ids)
        anchor_imgs = self.clusters[anchor_id]

        img_a_path, img_p_path = random.sample(anchor_imgs, 2)

        neg_id = random.choice(self.all_toad_ids)
        while neg_id == anchor_id or len(self.clusters[neg_id]) == 0:
            neg_id = random.choice(self.all_toad_ids)
        img_n_path = random.choice(self.clusters[neg_id])

        img_a = Image.open(img_a_path).convert("RGB")
        img_p = Image.open(img_p_path).convert("RGB")
        img_n = Image.open(img_n_path).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_p = self.transform(img_p)
            img_n = self.transform(img_n)

        return img_a, img_p, img_n, anchor_id, neg_id


class ToadGalleryDataset(Dataset):

    def __init__(self, image_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.transform = transform if transform is not None else get_default_transforms(is_train=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path), os.path.basename(str(path))

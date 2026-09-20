import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(current_dir))

from dataset import (
    ToadIdentityBatchDataset,
    PKBatchSampler,
    load_clusters_from_toad_id,
    get_default_transforms
)
from model import ToadMetricEmbeddingNet, BatchHardCosineTripletLoss, CosineTripletLoss

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def train_siamese_network(
    dataset="throat",
    mode="rgb",
    backbone="convnext_tiny",
    epochs=25,
    batch_size=32,
    p_classes=8,
    k_samples=4,
    lr=1e-4,
    margin=0.4,
    embedding_dim=256,
    batches_per_epoch=50,
    clusters_dir=None,
    test_dir=None
):
    dataset_name = "Throat + Belly" if dataset == "01-02_Throat_Belly_pipeline" else "Throat Only"
    print(f"\n=======================================================")
    print(f"🐸 Starting Siamese Metric Learning ({dataset_name} - {mode.upper()}) | Backbone: {backbone}")
    print(f"⚙️ Target Device: {device} | Epochs: {epochs} | P={p_classes}, K={k_samples} (Batch: {p_classes * k_samples})")
    print(f"🎯 Cosine Margin: {margin} | Input Resolution: 224x224")
    print(f"=======================================================\n")

    # Directories
    if clusters_dir is None:
        if dataset == "01-02_Throat_Belly_pipeline":
            clusters_dir = project_root / "03_2_Siamese_network" / "data" / "train" / "rgb_throat_belly_id"
        elif mode == "gray":
            clusters_dir = project_root / "03_2_Siamese_network" / "data" / "train" / "gray_toad_id"
        else:
            clusters_dir = project_root / "03_2_Siamese_network" / "data" / "train" / "rgb_toad_id"
    else:
        clusters_dir = Path(clusters_dir)

    if test_dir is None:
        if dataset == "01-02_Throat_Belly_pipeline":
            test_dir = project_root / "03_2_Siamese_network" / "data" / "test" / "rgb_throat_belly_test"
        elif mode == "gray":
            test_dir = project_root / "03_2_Siamese_network" / "data" / "test" / "gray_test"
        else:
            test_dir = project_root / "03_2_Siamese_network" / "data" / "test" / "rgb_test"
    else:
        test_dir = Path(test_dir)

    test_images = set(os.listdir(test_dir)) if test_dir.exists() else set()
    print(f"🔒 Strictly excluding {len(test_images)} test images from training to prevent data leakage.")

    sub_dir = f"{mode}_throat_belly" if dataset == "01-02_Throat_Belly_pipeline" else mode
    checkpoints_dir = project_root / "03_2_Siamese_network" / "checkpoints" / sub_dir
    results_dir = project_root / "results" / sub_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Ground Truth Clusters
    clusters = load_clusters_from_toad_id(toad_id_dir=clusters_dir, exclude_test_images=test_images)
    if not clusters:
        raise FileNotFoundError(f"❌ Error: No identity clusters found in '{clusters_dir}' or associated manifest.")

    total_images_in_clusters = sum(len(imgs) for imgs in clusters.values())
    total_multi_toads = sum(1 for imgs in clusters.values() if len(imgs) >= 2)
    print(f"📊 Dataset Loaded from '{clusters_dir}':")
    print(f"   - {len(clusters)} unique Toad IDs ({total_multi_toads} multi-sighting, {len(clusters) - total_multi_toads} singletons)")
    print(f"   - {total_images_in_clusters} total images assigned to identity clusters.")

    # Stratified train/val split
    multi_ids = [k for k, imgs in clusters.items() if len(imgs) >= 2]
    single_ids = [k for k, imgs in clusters.items() if len(imgs) == 1]
    
    np.random.seed(42)
    np.random.shuffle(multi_ids)
    np.random.shuffle(single_ids)

    n_val_multi = max(4, int(0.2 * len(multi_ids)))
    n_val_single = int(0.2 * len(single_ids))
    val_ids = set(multi_ids[:n_val_multi] + single_ids[:n_val_single])
    train_ids = set(multi_ids[n_val_multi:] + single_ids[n_val_single:])

    train_clusters = {k: clusters[k] for k in train_ids}
    val_clusters = {k: clusters[k] for k in val_ids}
    print(f"📊 Train Partition: {len(train_clusters)} Toad IDs ({sum(1 for imgs in train_clusters.values() if len(imgs) >= 2)} multi-sightings, {sum(len(imgs) for imgs in train_clusters.values())} images).")
    print(f"📊 Val Partition:   {len(val_clusters)} Toad IDs ({sum(1 for imgs in val_clusters.values() if len(imgs) >= 2)} multi-sightings, {sum(len(imgs) for imgs in val_clusters.values())} images).")

    # 2. Datasets & Batch-Hard Samplers
    train_dataset = ToadIdentityBatchDataset(train_clusters, transform=get_default_transforms(is_train=True))
    val_dataset = ToadIdentityBatchDataset(val_clusters, transform=get_default_transforms(is_train=False))

    train_sampler = PKBatchSampler(train_dataset.label_to_indices, p=p_classes, k=k_samples, num_batches=batches_per_epoch)
    val_sampler = PKBatchSampler(val_dataset.label_to_indices, p=max(4, p_classes // 2), k=k_samples, num_batches=max(15, batches_per_epoch // 3))

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    # 3. Model, Loss, Optimizer
    model = ToadMetricEmbeddingNet(backbone_name=backbone, embedding_dim=embedding_dim).to(device)
    criterion = BatchHardCosineTripletLoss(margin=margin)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_sim_pos": [], "train_sim_neg": [],
        "val_sim_pos": [], "val_sim_neg": []
    }

    # 4. Training Loop with Online Hard-Negative Mining
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_sim_pos = 0.0
        running_sim_neg = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train Batch-Hard]")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embeddings = model(imgs)
            loss, sim_p, sim_n = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_sim_pos += sim_p
            running_sim_neg += sim_n
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sim+": f"{sim_p:.2f}",
                "sim-": f"{sim_n:.2f}"
            })

        scheduler.step()

        train_loss = running_loss / max(num_batches, 1)
        train_sim_p = running_sim_pos / max(num_batches, 1)
        train_sim_n = running_sim_neg / max(num_batches, 1)

        # Validation Step
        model.eval()
        v_loss, v_sim_p, v_sim_n, v_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                embeddings = model(imgs)
                loss, sim_p, sim_n = criterion(embeddings, labels)
                v_loss += loss.item()
                v_sim_p += sim_p
                v_sim_n += sim_n
                v_batches += 1

        val_loss = v_loss / max(v_batches, 1)
        val_sim_p = v_sim_p / max(v_batches, 1)
        val_sim_n = v_sim_n / max(v_batches, 1)

        elapsed = time.time() - t0
        print(f" Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} (Sim+: {train_sim_p:.2f}, Sim-: {train_sim_n:.2f}) | Val Loss: {val_loss:.4f} (Sim+: {val_sim_p:.2f}, Sim-: {val_sim_n:.2f})")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sim_pos"].append(train_sim_p)
        history["train_sim_neg"].append(train_sim_n)
        history["val_sim_pos"].append(val_sim_p)
        history["val_sim_neg"].append(val_sim_n)

        # Checkpoint Saving
        checkpoint_path = checkpoints_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": embedding_dim,
            "mode": mode,
            "val_loss": val_loss
        }, checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoints_dir / "best_siamese_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "embedding_dim": embedding_dim,
                "mode": mode,
                "val_loss": val_loss
            }, best_path)
            print(f"   🌟 New best model saved (Val Loss: {val_loss:.4f})")

    # 5. Plot and Save Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["train_loss"], label="Train Batch-Hard Loss", color="#2563eb", lw=2)
    ax1.plot(history["val_loss"], label="Val Batch-Hard Loss", color="#dc2626", lw=2, linestyle="--")
    ax1.set_title("Batch-Hard Triplet Loss vs Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_sim_pos"], label="Train Sim (Hardest Positive)", color="#16a34a", lw=2)
    ax2.plot(history["train_sim_neg"], label="Train Sim (Hardest Negative)", color="#ea580c", lw=2)
    ax2.plot(history["val_sim_pos"], label="Val Sim (Hardest Positive)", color="#16a34a", lw=1.5, linestyle="--")
    ax2.plot(history["val_sim_neg"], label="Val Sim (Hardest Negative)", color="#ea580c", lw=1.5, linestyle="--")
    ax2.set_title("Hard Pair Cosine Separation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cosine Similarity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = results_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()

    with open(results_dir / "training_metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training Complete! Best model saved to: {checkpoints_dir / 'best_siamese_model.pt'}")
    print(f"📊 Training curves saved to: {curve_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Siamese Metric Learning: Batch-Hard Mining")
    parser.add_argument("--dataset", type=str, choices=["throat", "01-02_Throat_Belly_pipeline"], default="throat", help="Dataset target: throat or 01-02_Throat_Belly_pipeline (default: throat)")
    parser.add_argument("--mode", type=str, choices=["rgb", "gray"], default="rgb")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Vision backbone (default: convnext_tiny)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--p_classes", type=int, default=8)
    parser.add_argument("--k_samples", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--clusters_dir", type=str, default=None, help="Path to toad_id clusters folder")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to test images folder to exclude")
    args = parser.parse_args()

    train_siamese_network(
        dataset=args.dataset,
        mode=args.mode,
        backbone=args.backbone,
        epochs=args.epochs,
        p_classes=args.p_classes,
        k_samples=args.k_samples,
        lr=args.lr,
        margin=args.margin,
        embedding_dim=args.embedding_dim,
        batches_per_epoch=args.batches,
        clusters_dir=args.clusters_dir,
        test_dir=args.test_dir
    )

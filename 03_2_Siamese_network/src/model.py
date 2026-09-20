import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ToadMetricEmbeddingNet(nn.Module):
    """
    Deep Metric Learning Network for Toad Ventral Biometric Re-Identification.
    
    Backbone: ConvNeXt-Tiny (standard computer vision foundation model).
    Embedding Head: Multi-layer non-linear projection yielding L2-normalized metric vectors.
    """
    def __init__(self, backbone_name="convnext_tiny", embedding_dim=256, pretrained=True, freeze_backbone_epochs=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone_name
        
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            in_features = self.backbone.num_features
        except Exception as e:
            print(f"⚠️ Could not load {backbone_name} via timm ({e}). Falling back to torchvision convnext_tiny...")
            import torchvision.models as models
            self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()

        # Projection Head: Non-linear MLP with Batch Normalization
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def extract_features(self, x):
        """Extracts raw backbone representation."""
        return self.backbone(x)

    def forward(self, x):
        """
        Forward pass producing unit L2-normalized embedding vectors.
        Output shape: [Batch_Size, embedding_dim]
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        
        embeddings = self.projection_head(features)
        # Strict L2 normalization to project onto unit hypersphere S^(d-1)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings


class CosineTripletLoss(nn.Module):
    """
    Triplet Margin Loss based on Cosine Similarity:
    Encourages: CosineSim(Anchor, Positive) > CosineSim(Anchor, Negative) + margin
    
    Loss = max(0, CosineSim(Anchor, Negative) - CosineSim(Anchor, Positive) + margin)
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Both inputs are L2-normalized, so dot product equals cosine similarity
        sim_ap = torch.sum(anchor * positive, dim=1)
        sim_an = torch.sum(anchor * negative, dim=1)

        losses = F.relu(sim_an - sim_ap + self.margin)
        return losses.mean(), sim_ap.mean().item(), sim_an.mean().item()


class BatchHardCosineTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss with Cosine Distance.
    
    For every anchor image i in the mini-batch:
    - Hardest positive: p* = argmin_{j: y_j == y_i} cos(z_i, z_j)
    - Hardest negative: n* = argmax_{k: y_k != y_i} cos(z_i, z_k)
    
    Loss = mean( max(0, cos(z_i, z_n*) - cos(z_i, z_p*) + margin) )
    
    Forces extreme intra-individual compactness and large inter-individual separation.
    """
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        sim_mat = torch.mm(embeddings, embeddings.t())  # [B, B]
        B = embeddings.size(0)

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t())  # [B, B]
        neg_mask = (labels != labels.t())  # [B, B]

        # Exclude self-matching from positive mask
        pos_mask.fill_diagonal_(False)

        losses = []
        hardest_pos_sims = []
        hardest_neg_sims = []

        for i in range(B):
            pos_sims_i = sim_mat[i][pos_mask[i]]
            if len(pos_sims_i) == 0:
                continue
            hardest_pos = torch.min(pos_sims_i)

            neg_sims_i = sim_mat[i][neg_mask[i]]
            if len(neg_sims_i) == 0:
                continue
            hardest_neg = torch.max(neg_sims_i)

            loss_i = F.relu(hardest_neg - hardest_pos + self.margin)
            losses.append(loss_i)
            hardest_pos_sims.append(hardest_pos.item())
            hardest_neg_sims.append(hardest_neg.item())

        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device), 0.0, 0.0

        total_loss = torch.stack(losses).mean()
        avg_pos = sum(hardest_pos_sims) / len(hardest_pos_sims) if hardest_pos_sims else 0.0
        avg_neg = sum(hardest_neg_sims) / len(hardest_neg_sims) if hardest_neg_sims else 0.0
        return total_loss, avg_pos, avg_neg


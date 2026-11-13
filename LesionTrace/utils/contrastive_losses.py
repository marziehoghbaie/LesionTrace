import torch
import torch.nn.functional as F


def contrastive_loss_tokenwise(f1, f2, temperature=0.1):
    """
    f1, f2: (B, C, N) features
    Token-wise contrastive loss: encourages f1[:,:,i] ~ f2[:,:,i]
    """
    B, C, N = f1.shape

    # Normalize
    f1 = F.normalize(f1, dim=1)  # across channels
    f2 = F.normalize(f2, dim=1)

    # Flatten batch and tokens -> (B*N, C)
    f1_flat = f1.permute(0, 2, 1).reshape(B*N, C)
    f2_flat = f2.permute(0, 2, 1).reshape(B*N, C)

    # Compute similarity matrix (BN x BN)
    logits = torch.matmul(f1_flat, f2_flat.T) / temperature

    # Positive pairs: diagonal alignment (same token index)
    labels = torch.arange(B*N, device=f1.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i + loss_j)




def contrastive_loss_global(f1, f2, temperature=0.1):
    """
    f1, f2: (B, C, N) feature maps
    Computes global contrastive loss encouraging f1 ~ f2 similarity.
    """

    # Global average pooling across spatial tokens
    f1 = f1.mean(dim=-1)  # (B, 768)
    f2 = f2.mean(dim=-1)  # (B, 768)

    # Normalize embeddings
    f1 = F.normalize(f1, dim=-1)
    f2 = F.normalize(f2, dim=-1)

    # Compute similarity matrix (B x B)
    logits = torch.matmul(f1, f2.T) / temperature  # cosine similarity scaled

    # Contrastive targets (diagonal = positive pairs)
    labels = torch.arange(f1.size(0), device=f1.device)

    # Symmetric InfoNCE loss
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i + loss_j)

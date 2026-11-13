import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalAngleLoss(nn.Module):
    def __init__(self, feature_dim, logger, args):
        """
        *** A bias-free logistic-regression-style projection applied to feature differences,
         trained to represent ordered change instead of class probability.
        *** learning a signed projection of feature displacement onto a learned disease progression axis, with margin-based tolerance for stable cases.
         """
        super().__init__()
        self.d = nn.Parameter(torch.randn(feature_dim))
        logger.info('squash projection with tanh to keep it in [-1, +1] range')
        self.drop_out = args.drop_out
        self.add_noise = args.add_noise

        if self.drop_out:
            self.dropout = nn.Dropout(p=0.3)

    def forward(self, f1, f2, labels):
        t = labels.float() - 1  # -1, 0, +1

        d_norm = F.normalize(self.d, p=2, dim=0)

        if self.add_noise:
            f1 = f1 + 0.01 * torch.randn_like(f1)
            f2 = f2 + 0.01 * torch.randn_like(f2)
        if self.drop_out:
            f1 = self.dropout(f1)
            f2 = self.dropout(f2)

        v = f2 - f1
        s = torch.matmul(v, d_norm)
        s_tanh = torch.tanh(s)

        loss = (s_tanh - t) ** 2

        # Margin for stable class: only penalize if projection is outside [-margin, +margin]
        # This tolerates natural feature variation in stable-but-lesioned samples
        margin = 0.2
        stable_mask = (labels == 1)
        if stable_mask.sum() > 0:
            stable_loss = loss[stable_mask]
            # Zero out loss for stable samples already within the margin
            within_margin = (s_tanh[stable_mask].abs() < margin)
            stable_loss = stable_loss * (~within_margin).float()
            loss = torch.cat([loss[~stable_mask], stable_loss])

        return loss.mean()



class DirectionalAngleLossDeadMargin(nn.Module):
    def __init__(self, feature_dim, logger, args):
        super().__init__()
        self.d = nn.Parameter(torch.randn(feature_dim))
        logger.info('squash projection with tanh to keep it in [-1, +1] range')
        self.drop_out = args.drop_out
        self.add_noise = args.add_noise
        self.epsilon = args.epsilon
        if self.drop_out:
            self.dropout = nn.Dropout(p=0.3)

    def forward(self, f1, f2, labels):
        """
        f1, f2: [batch_size, feature_dim]
        labels: [batch_size], with values 0=reduce, 1=stable, 2=increase
        """
        # map to signed targets: reduced -> -1, stable -> 0, increased -> +1
        t = labels.float() - 1  # 0->-1, 1->0, 2->+1

        # normalize change axis
        d_norm = F.normalize(self.d, p=2, dim=0)  # [feature_dim]

        if self.add_noise:
            n1 = 0.01 * torch.randn_like(f1)
            f1 = f1 + n1
            n2 = 0.01 * torch.randn_like(f2)
            f2 = f2 + n2
        if self.drop_out:
            f1 = self.dropout(f1)
            f2 = self.dropout(f2)
        # difference vector
        v = f2 - f1  # [batch_size, feature_dim]

        # projection
        s = torch.matmul(v, d_norm)

        # stable margin => epsilon

        loss = torch.where(
            labels == 1,
            F.relu(torch.abs(s) - self.epsilon) ** 2,
            (torch.tanh(s) - t) ** 2
        )

        return loss.mean()


class DirectionalAngleLossMag_Dir(nn.Module):
    def __init__(self, feature_dim, logger, args):
        super().__init__()
        self.d = nn.Parameter(torch.randn(feature_dim))
        logger.info('squash projection with tanh to keep it in [-1, +1] range')
        self.drop_out = args.drop_out
        self.add_noise = args.add_noise

        if self.drop_out:
            self.dropout = nn.Dropout(p=0.3)

    def forward(self, f1, f2, labels):
        """
        f1, f2: [batch_size, feature_dim]
        labels: [batch_size], with values 0=reduce, 1=stable, 2=increase
        """
        # map to signed targets: reduced -> -1, stable -> 0, increased -> +1
        t = labels.float() - 1  # 0->-1, 1->0, 2->+1

        # normalize change axis
        d_norm = F.normalize(self.d, p=2, dim=0)  # [feature_dim]

        if self.add_noise:
            n1 = 0.01 * torch.randn_like(f1)
            f1 = f1 + n1
            n2 = 0.01 * torch.randn_like(f2)
            f2 = f2 + n2
        if self.drop_out:
            f1 = self.dropout(f1)
            f2 = self.dropout(f2)
        # difference vector
        v = f2 - f1  # [batch_size, feature_dim]

        # projection
        s = torch.matmul(v, d_norm)

        # magnitude
        mag = torch.abs(s)

        # direction loss (only for non-stable)
        dir_loss = ((torch.tanh(s[labels != 1]) - t[labels != 1]) ** 2).mean()

        # magnitude loss
        mag_target = (labels != 1).float()  # 0 for stable, 1 for change
        mag_loss = (mag - mag_target) ** 2

        loss = dir_loss + 0.5 * mag_loss.mean()

        return loss.mean()



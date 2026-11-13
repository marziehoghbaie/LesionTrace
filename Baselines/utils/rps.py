import torch
import torch.nn as nn
import torch.nn.functional as F



class saRPSLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, preds, target):
        # Convert target to one-hot if needed
        if target.dim() == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        # Compute cumulative sums
        cumul_preds = torch.cumsum(preds, dim=1)
        cumul_target = torch.cumsum(target, dim=1)

        # Compute absolute differences
        abs_diff = torch.abs(cumul_preds - cumul_target)
        sum_abs = abs_diff.sum(dim=1)
        loss = (sum_abs ** 2) / (self.num_classes - 1)

        return self._reduce(loss)

    def _reduce(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def sa_rps_metric(preds, target, num_classes):
    return saRPSLoss(num_classes, reduction='mean')(preds, target)

if __name__ == "__main__":

    # For a 3-class problem
    num_classes = 3
    loss_fn = saRPSLoss(num_classes, reduction='mean')

    # Predictions (softmax outputs)
    preds = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])

    # Targets as class indices
    target = torch.tensor([0, 1])

    loss = loss_fn(preds, target)
    print("sa-RPS Loss:", loss.item())

    # As metric
    metric_val = sa_rps_metric(preds, target, num_classes)
    print("sa-RPS Metric:", metric_val.item())

    # Test RPS for Page 4 example
    num_classes = 3
    y_true = torch.tensor([0])  # Class index 0 (one-hot: [1,0,0])
    pred1 = torch.tensor([[0.25, 0.75, 0.0]])
    pred2 = torch.tensor([[0.25, 0.0, 0.75]])


    # Test RPS symmetry issue
    y_true_mid = torch.tensor([1])  # Class index 1 (one-hot: [0,1,0])
    pred_sym = torch.tensor([[0.3, 0.4, 0.3]])
    pred_asym = torch.tensor([[0.45, 0.5, 0.05]])

    # Test sa-RPS correction
    sarps_sym = sa_rps_metric(pred_sym, y_true_mid, num_classes).item()
    sarps_asym = sa_rps_metric(pred_asym, y_true_mid, num_classes).item()

    print(f"sa-RPS Symmetric: {sarps_sym:.4f} (Expected: 0.1800)")
    print(f"sa-RPS Asymmetric: {sarps_asym:.4f} (Expected: 0.1250)")

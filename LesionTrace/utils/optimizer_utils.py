import torch
import torch.optim as optim
from adan_pytorch import Adan
from torch.nn import Module


def create_optimizer(train_config: dict, model: Module) -> optim.Optimizer:
    """
    Create a PyTorch optimizer based on the training configuration.

    Supported optimizers: SGD, Adam, AdamW, Adan.

    Args:
        train_config (dict): Training configuration with keys:
            - optimizer: 'sgd', 'adam', 'adamw', or 'adan'
            - init_lr: initial learning rate
            - momentum: momentum (for SGD)
            - weight_decay: weight decay
        model (torch.nn.Module): Model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    optimizer_type = train_config.get("optimizer", "adam").lower()
    lr = train_config.get("init_lr", 1e-3)
    weight_decay = train_config.get("weight_decay", 0.0)
    momentum = train_config.get("momentum", 0.9)

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adan":
        optimizer = Adan(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.1, 0.1, 0.001),  # as described in Adan paper
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

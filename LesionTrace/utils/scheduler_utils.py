import math
import torch
import torch.optim as optim


# ------------------------
# LR Utilities
# ------------------------
def change_lr_on_optimizer(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate of all param groups in the optimizer."""
    for g in optimizer.param_groups:
        g["lr"] = lr


# ------------------------
# Custom Schedulers
# ------------------------
class CosineScheduler:
    """Cosine decay scheduler over epochs."""

    def __init__(self, lr_ori: float, epochs: int):
        self.lr_ori = lr_ori
        self.epochs = epochs

    def adjust_lr(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """Adjust LR based on cosine decay for current epoch."""
        ratio = 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))
        change_lr_on_optimizer(optimizer, self.lr_ori * ratio)


class InverseSquareRootScheduler:
    """
    Warmup followed by inverse square root decay scheduler.

    Usage:
        sch = InverseSquareRootScheduler(warmup_init_lr, warmup_end_lr, warmup_updates)
        sch.adjust_lr(optimizer, num_updates)
    """

    def __init__(self, warmup_init_lr: float, warmup_end_lr: float, warmup_updates: int = 4000):
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_updates = warmup_updates

        # Linear warmup step size
        self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / float(self.warmup_updates)
        # Decay factor for inverse sqrt
        self.decay_factor = self.warmup_end_lr * math.sqrt(self.warmup_updates)
        self.lr = self.warmup_init_lr

    def adjust_lr(self, optimizer: torch.optim.Optimizer, num_updates: int) -> None:
        """Adjust LR based on warmup + inverse square root decay."""
        if num_updates < self.warmup_updates:
            self.lr = self.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * (num_updates ** -0.5)

        change_lr_on_optimizer(optimizer, self.lr)


# ------------------------
# Scheduler Factory
# ------------------------
def create_scheduler(train_config: dict, optimizer: torch.optim.Optimizer, len_train_loader: int):
    """
    Create a learning rate scheduler based on training config.

    Args:
        train_config: Dictionary containing scheduler type and parameters.
        optimizer: Optimizer to wrap.
        len_train_loader: Number of batches per epoch.

    Returns:
        Scheduler object or None.
    """
    steps = train_config.get('step_coeff', 1) * len_train_loader
    opt_name = train_config.get('optimizer', 'adam')
    cycle_momentum = False if opt_name.lower() == 'adam' else True

    scheduler_type = train_config.get('scheduler', None)

    if scheduler_type == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=train_config['base_lr_cyclic'],
            max_lr=train_config['max_lr'],
            step_size_up=steps,
            mode='triangular2',
            cycle_momentum=cycle_momentum,
            verbose=False
        )
    elif scheduler_type == 'cosine_annealing':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            eta_min=1e-7
        )
    elif scheduler_type == 'cosine_scheduler':
        return CosineScheduler(train_config['init_lr'], train_config['num_epochs'])
    elif scheduler_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    elif scheduler_type == 'isr':
        return InverseSquareRootScheduler(
            warmup_init_lr=train_config['init_lr'],
            warmup_end_lr=train_config['max_lr'],
            warmup_updates=steps
        )
    elif scheduler_type == 'cosine_with_warmup':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            eta_min=train_config.get('eta_min', 1.0e-7),
            last_epoch=train_config.get('last_epoch', -1),
        )
    elif scheduler_type == 'cosine_annealing_warm_restarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=train_config['warmup_epochs'],
            T_mult=train_config.get('T_mult', 1),
            eta_min=train_config.get('eta_min', 1e-7),
            last_epoch=train_config.get('last_epoch', -1),
            verbose=False
        )
    else:
        return None

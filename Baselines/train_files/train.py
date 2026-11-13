import time
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from utils.utils import showLR, draw_results  # keep your helpers
from test_files.test import test, scoring


def train_val(
    train_config: dict,
    logger,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    train_loader,
    val_loader,
    scheduler,
    ckpt_saver,
    init_epoch: int = 0,
    late_fusion: bool = False,
    verbose: bool = True,
    log_interval: int = 100,
    max_epochs: Optional[int] = None,
) -> Tuple[torch.nn.Module, Optional[str]]:
    """
    Train/validate loop with mixed-precision and tensorboard logging.

    Args:
        train_config: training config dict (must contain 'use_gpu' and 'num_epochs' etc.)
        logger: logging object
        model: PyTorch model
        optimizer: optimizer
        loss_fn: classification loss function (callable)
        train_loader, val_loader: DataLoaders
        scheduler: LR scheduler or None
        ckpt_saver: checkpoint saver with .save(save_dict, metric, metric_name)
        init_epoch: starting epoch index
        late_fusion: whether model expects (A,B) or a single input 'imgs'
        verbose: whether to print tqdm and logs
        log_interval: batches between detailed logging
        max_epochs: override num_epochs in config if set

    Returns:
        model (best state kept externally by ckpt_saver), best_checkpoint_path (or None)
    """

    writer = SummaryWriter(log_dir=ckpt_saver.save_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() and train_config.get("use_gpu", False) else "cpu")
    logger.info(f"Model will run on {device}")

    model = model.to(device)
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    best_checkpoint_path = None
    best_acc = 0.0
    num_epochs = train_config.get("num_epochs", 1)
    save_metric_name = 'f1'
    scheduler_type = train_config.get("scheduler", None)
    logger.info(f"Scheduler type: {scheduler_type}")
    logger.info(f"The best epoch is saved based on {save_metric_name}")
    logger.info("The F1 score is set to Macro by default, which is more suitable for imbalanced datasets.")

    running_all = len(train_loader.dataset)
    iters = len(train_loader) if len(train_loader) > 0 else 1

    epochs_loss, epochs_acc = [], []
    epochs_loss_val, epochs_acc_val = [], []

    for epoch in range(init_epoch, num_epochs):
        model.train()

        epoch_loss_sum = 0.0
        y_true = []
        y_pred = []

        epoch_start = time.time()
        for idx, batch in enumerate(train_loader):
            # zero grads
            optimizer.zero_grad(set_to_none=True)

            # move inputs & labels to device once
            labels = batch["label"].to(device, non_blocking=True)

            A = batch["A"].to(device, non_blocking=True)
            B = batch["B"].to(device, non_blocking=True)
            outputs = model(A, B)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            # accumulate metrics
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # compute loss (single scalar)
            loss = loss_fn(logits, labels)

            # backward + step (mixed precision)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # scheduler step: handle reduce_on_plateau separately later (after val)
            if scheduler is not None and scheduler_type != "reduce_on_plateau":
                # step per batch with epoch fraction if needed
                scheduler.step(epoch + idx / max(1, iters))

            # aggregation for epoch stats
            batch_size = labels.size(0)
            epoch_loss_sum += loss.item() * batch_size

            # tensorboard (batch)
            global_step = epoch * iters + idx
            writer.add_scalar("Loss/train/batch", loss.item(), global_step)
            # logging

            if idx % log_interval == 0:
                batch_loss = epoch_loss_sum / ((idx + 1) * batch_size)
                batch_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred) if len(y_true) > 0 else 0.0
                scores = scoring(y_true, y_pred) if len(y_true) > 0 else {"F1_score": 0.0}
                if verbose:
                    logger.info(f"✅ [Epoch {epoch}][Batch {idx}] LR={showLR(optimizer)} loss={batch_loss:.4f} bacc={batch_acc:.4f} F1={scores['F1_score']:.4f}")
        logger.info(f"This epoch took {time.time() - epoch_start:.1f}s")

        # epoch metrics (training)
        epoch_loss = epoch_loss_sum / running_all
        epoch_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred) if len(y_true) > 0 else 0.0
        epoch_scores = scoring(y_true, y_pred) if len(y_true) > 0 else {"F1_score": 0.0}
        epoch_f1 = epoch_scores["F1_score"]

        epochs_loss.append(epoch_loss)
        epochs_acc.append(epoch_acc)

        # Validation
        logger.info(f"⚠️ [INFO] Validation for epoch {epoch} starting...")
        val_start = time.time()
        val_metrics = test(val_loader, model, loss_fn, logger, late_fusion=late_fusion, complete=False, train_config=train_config)
        logger.info(f"[INFO] Validation done in {time.time() - val_start:.1f}s")

        val_acc = val_metrics["bacc"]
        val_f1 = val_metrics.get("F1_score", 0.0)
        val_loss = val_metrics["loss"]

        # scheduler for reduce_on_plateau (needs val metric)
        if scheduler is not None and scheduler_type == "reduce_on_plateau":
            # If scheduler expects metric, supply val_loss or val_acc depending on config
            monitor = val_loss if getattr(scheduler, "_metric", None) is None else val_acc
            try:
                scheduler.step(monitor)
            except Exception:
                scheduler.step(val_loss)

        # Save best checkpoint (based on chosen metric)
        metric_value = val_acc if save_metric_name == "bacc" else val_f1
        if metric_value > best_acc:
            best_acc = metric_value
            if best_acc > train_config.get("save_threshold", 0.0):
                save_dict = {
                    "epoch_idx": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                best_checkpoint_path = ckpt_saver.save(save_dict, metric_value, metric=save_metric_name)
                logger.info(f"🔥 [INFO] Checkpoint saved at epoch {epoch} metric={save_metric_name}:{metric_value:.4f}")

        # epoch-level tensorboard
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("ACC/train", epoch_acc, epoch)
        writer.add_scalar("F1/train", epoch_f1, epoch)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("ACC/val", val_acc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        epochs_loss_val.append(val_loss)
        epochs_acc_val.append(val_acc)

        logger.info(
            f"Epoch {epoch} summary: train_loss={epoch_loss:.4f} train_bacc={epoch_acc:.4f} val_loss={val_loss:.4f} val_bacc={val_acc:.4f} lr={showLR(optimizer)}"
        )

    # teardown
    writer.close()
    draw_results([epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val], save_path=ckpt_saver.save_dir)
    return model, best_checkpoint_path

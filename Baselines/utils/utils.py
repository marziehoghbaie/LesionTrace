from __future__ import absolute_import, division, print_function

import hashlib
import inspect
import json
import logging
import math
import os
import sys
from copy import deepcopy
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score
)

from thop import profile
import torch
from sklearn.preprocessing import LabelBinarizer
from torchvision.transforms import Compose, ToPILImage, ToTensor, Normalize, Resize, Grayscale
from torchvision.transforms.functional import InterpolationMode
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, balanced_accuracy_score,
    roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import LabelBinarizer
from utils.rps import sa_rps_metric


# ------------------------
# Path helpers
# ------------------------
def add_project_paths(levels_up: int = 3) -> None:
    """Insert parent dirs up to levels_up into sys.path (for local imports)."""
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    sys.path.insert(0, current_dir)
    for _ in range(levels_up):
        current_dir = os.path.dirname(current_dir)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)


add_project_paths(3)

# ------------------------

from bstrap import boostrapping_CI

# ------------------------
# Constants / simple transforms
# ------------------------
TRANSFORM_RAW_IMG = Compose([
    ToTensor(),
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
])


# ------------------------
# Print trainable parameters
# ------------------------
def print_trainable_parameters(model, logger=None):
    """
    Logs the number of trainable and total parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = 100 * trainable_params / total_params

    msg = f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable %: {trainable_percent:.2f}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ------------------------
# Logging helper
# ------------------------
def create_logger(save_path: str, name: str = "") -> logging.Logger:
    """Create file + console logger writing to save_path/log_{name}.txt."""
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"log_{name}.txt")
    logger = logging.getLogger(f"utils_logger_{name}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(filename, mode="a+")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


# ------------------------
# Bootstrapped CI
# ------------------------
def change_bootstrap_CI(path: str, nbr_runs: int = 1000, file_name: str = 'results.csv', external_test=False) -> None:
    """
    Compute bootstrap confidence intervals for multiple metrics and append to log file.
    Expects a CSV at {path}/results.csv with columns: y_probs, y_pred, y_true.
    """
    log_fp = os.path.join(path, "log_.txt")

    df = pd.read_csv(os.path.join(path, file_name))

    # --- Parse y_probs column ---
    def parse_probs(x):
        if isinstance(x, str):
            s = x.strip("[]")
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            return [float(p) for p in parts if p]
        elif isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        else:
            return [float(x)]

    df["y_probs"] = df["y_probs"].apply(parse_probs)

    if external_test:
        df["y_probs"] = df["y_probs"].apply(lambda x: np.array(x)[:3])

    # --- Metric Bootstrapping ---
    metrics_to_eval = {
        "f1_score": (f1_score, {"average": "weighted"}),
        "precision_score": (precision_score, {"average": "weighted"}),
        "recall_score": (recall_score, {"average": "weighted"}),
        "balanced_accuracy_score": (balanced_accuracy_score, {}),
        "accuracy_score": (accuracy_score, {}),
        "matthews_corrcoef": (matthews_corrcoef, {}),
        "specificity": (specificity, {}),
        "rps": (sa_rps_metric, {}),
        "average_precision_score": (average_precision_score, {}),
        "roc_auc_score": (roc_auc_score, {'multi_class': 'ovr'}),
        "cohen_kappa_score": (cohen_kappa_score, {}),

    }

    results = {}
    for metric_name, (metric_func, kwargs) in metrics_to_eval.items():
        # print(metric_name, flush=True)
        results[metric_name] = boostrapping_CI(
            metric_func,
            df,
            nbr_runs=nbr_runs,
            metric_type=metric_name,
            **kwargs,
        )
        # print(metric_name, results[metric_name])

    # --- ROC AUC (binary or multi-class) ---
    unique_classes = np.unique(df["y_true"])
    if len(unique_classes) == 2:
        df["y_probs"] = df["y_probs"].apply(lambda x: x[1] if len(x) > 1 else x[0])
        df["y_onehot_test"] = df["y_true"]
    else:
        lb = LabelBinarizer().fit(list(df["y_true"].values))
        df["y_onehot_test"] = list(lb.transform(list(df["y_true"].values)))

    # --- Write Results ---
    with open(log_fp, "a") as f:
        f.write("\nBootstrap CI results:\n")
        for m, res in results.items():
            avg_metric = res["avg_metric"]
            ci_lb = res["metric_ci_lb"]
            ci_ub = res["metric_ci_ub"]
            f.write(f"{m:20s}: {avg_metric:.3f} ({ci_lb:.3f}-{ci_ub:.3f})\n")
            print(f"{m:20s}: {avg_metric:.3f} ({ci_lb:.3f}-{ci_ub:.3f})")

    return log_fp


def specificity(y_true, y_pred, eps=1e-6):
    """Calculate average specificity for multi-class classification."""
    cm = confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return float((TN / (TN + FP + eps)).mean())



def change_bootstrap_CI(path: str, nbr_runs: int = 1000, file_name: str = 'results.csv', external_test=False) -> None:
    """
    Compute bootstrap confidence intervals for multiple metrics and append to log file.
    Expects a CSV at {path}/results.csv with columns: y_probs, y_pred, y_true.
    """
    log_fp = os.path.join(path, "log_.txt")

    df = pd.read_csv(os.path.join(path, file_name))

    # --- Parse y_probs column ---
    def parse_probs(x):
        if isinstance(x, str):
            s = x.strip("[]")
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            return [float(p) for p in parts if p]
        elif isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        else:
            return [float(x)]

    df["y_probs"] = df["y_probs"].apply(parse_probs)

    if external_test:
        df["y_probs"] = df["y_probs"].apply(lambda x: np.array(x)[:3])

    # --- Metric Bootstrapping ---
    metrics_to_eval = {
        "f1_score": (f1_score, {"average": "macro"}),
        "precision_score": (precision_score, {"average": "weighted"}),
        "recall_score": (recall_score, {"average": "weighted"}),
        "balanced_accuracy_score": (balanced_accuracy_score, {}),
        "accuracy_score": (accuracy_score, {}),
        "matthews_corrcoef": (matthews_corrcoef, {}),
        "specificity": (specificity, {}),
        "rps": (sa_rps_metric, {}),
        "average_precision_score": (average_precision_score, {}),
        "roc_auc_score": (roc_auc_score, {'multi_class': 'ovr'}),
        "cohen_kappa_score": (cohen_kappa_score, {}),

    }

    results = {}
    for metric_name, (metric_func, kwargs) in metrics_to_eval.items():
        # print(metric_name, flush=True)
        results[metric_name] = boostrapping_CI(
            metric_func,
            df,
            nbr_runs=nbr_runs,
            metric_type=metric_name,
            **kwargs,
        )
        # print(metric_name, results[metric_name])

    # --- ROC AUC (binary or multi-class) ---
    unique_classes = np.unique(df["y_true"])
    if len(unique_classes) == 2:
        df["y_probs"] = df["y_probs"].apply(lambda x: x[1] if len(x) > 1 else x[0])
        df["y_onehot_test"] = df["y_true"]
    else:
        lb = LabelBinarizer().fit(list(df["y_true"].values))
        df["y_onehot_test"] = list(lb.transform(list(df["y_true"].values)))

    # --- Write Results ---
    with open(log_fp, "a") as f:
        f.write("\nBootstrap CI results:\n")
        for m, res in results.items():
            avg_metric = res["avg_metric"]
            ci_lb = res["metric_ci_lb"]
            ci_ub = res["metric_ci_ub"]
            f.write(f"{m:20s}: {avg_metric:.3f} ({ci_lb:.3f}-{ci_ub:.3f})\n")
            print(f"{m:20s}: {avg_metric:.3f} ({ci_lb:.3f}-{ci_ub:.3f})")

    return log_fp

# ------------------------
# Model utils
# ------------------------
def get_model_hash(model: torch.nn.Module) -> str:
    """Compute MD5 over model parameters (CPU numpy bytes)."""
    # To avoid allocating one giant array, hash piecewise
    md5 = hashlib.md5()
    for p in model.parameters():
        md5.update(p.detach().cpu().numpy().tobytes())
    return md5.hexdigest()


def get_model_size(model: torch.nn.Module) -> float:
    """Return model size in megabytes (params + buffers)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def num_model_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters (int)."""
    return sum(p.numel() for p in model.parameters())


# ------------------------
# Numeric helpers
# ------------------------
def calculateNorm2(model: torch.nn.Module) -> float:
    """Return the 2-norm of all parameters (useful for diagnostics)."""
    total = 0.0
    for p in model.parameters():
        total += (p.data.norm(2).item()) ** 2
    return math.sqrt(total)


def showLR(optimizer: torch.optim.Optimizer) -> float:
    """Return learning rate of first parameter group."""
    return float(optimizer.param_groups[0]["lr"])


class AverageMeter:
    """Utility to accumulate and compute average values (e.g., losses)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# ------------------------
# I/O utilities
# ------------------------
def read_txt_lines(filepath: str):
    assert os.path.isfile(filepath), f"File not found: {filepath}"
    with open(filepath, "r") as fh:
        return fh.read().splitlines()


def save_as_json(d: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as fh:
        json.dump(d, fh, indent=4, sort_keys=True)


def load_json(json_fp: str) -> dict:
    assert os.path.isfile(json_fp), f"File not found: {json_fp}"
    with open(json_fp, "r") as fh:
        return json.load(fh)


def save2npz(filename: str, data) -> None:
    assert data is not None
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename, data=data)


# ------------------------
# Checkpointing
# ------------------------
class CheckpointSaver:
    def __init__(
            self,
            save_dir: str,
            checkpoint_fn: str = "ckpt.pth.tar",
            best_fn: str = "ckpt.best.pth.tar",
            best_step_fn: str = "ckpt.best.step{}.pth.tar",
            save_best_step: bool = False,
            lr_steps: Optional[list] = None,
    ):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoint_fn = checkpoint_fn
        self.best_fn = best_fn
        self.best_step_fn = best_step_fn
        self.save_best_step = save_best_step
        self.lr_steps = lr_steps or []
        self.current_best = -float("inf")
        if self.save_best_step and not self.lr_steps:
            raise ValueError("lr_steps must be provided when save_best_step=True")
        if self.save_best_step:
            self.best_for_stage = [-float("inf")] * (len(self.lr_steps) + 1)

    def save(self, save_dict: dict, current_perf: float, epoch: int = -1, metric: str = "acc") -> str:
        """Save checkpoint and optionally store best/step-best files. Returns checkpoint path."""
        # Compose checkpoint filename
        checkpoint_name = f"current_val_{metric}_{current_perf:.4f}_ckpt.pth.tar"
        checkpoint_fp = os.path.join(self.save_dir, checkpoint_name)

        # update best
        is_best = current_perf > self.current_best
        if is_best:
            self.current_best = current_perf
            best_name = f"best_val_{metric}_{current_perf:.4f}_ckpt.pth.tar"
            best_fp = os.path.join(self.save_dir, best_name)
            torch.save(save_dict, best_fp)
        save_dict["best_prec"] = self.current_best

        # step best logic
        if self.save_best_step:
            if epoch < 0:
                raise ValueError("epoch must be >= 0 when save_best_step=True")
            sidx = sum(epoch >= s for s in self.lr_steps)
            is_best_stage = current_perf > self.best_for_stage[sidx]
            if is_best_stage:
                self.best_for_stage[sidx] = current_perf
                stage_fp = os.path.join(self.save_dir, self.best_step_fn.format(sidx))
                torch.save(save_dict, stage_fp)
            save_dict["best_prec_per_stage"] = self.best_for_stage

        # always save current checkpoint
        torch.save(save_dict, checkpoint_fp)
        print(f"Checkpoint saved at {checkpoint_fp}")
        return checkpoint_fp

    def set_best_from_ckpt(self, ckpt_dict: dict):
        self.current_best = ckpt_dict.get("best_prec", self.current_best)
        self.best_for_stage = ckpt_dict.get("best_prec_per_stage", getattr(self, "best_for_stage", None))


# ------------------------
# Model loading / conversion helpers
# ------------------------
def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor, num_tokens: int = 1,
                     gs_new: Tuple[int, int] = ()) -> torch.Tensor:
    """Resize position embeddings (ViT style) to new grid size."""
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = nn.functional.interpolate(posemb_grid, size=gs_new, mode="bicubic", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def load_pretrained_weights_vfm(model: torch.nn.Module, pretrained_weights: str, checkpoint_key: Optional[str],
                                model_name: str, patch_size: int) -> None:
    """
    Load pretrained weights from file or from official URLs (DINO / supervised).
    This function will resize pos_embed if required and load weights with strict=False.
    """
    if os.path.isfile(pretrained_weights):
        state = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state:
            state = state[checkpoint_key]

        # sanitize keys (module/backbone prefixes)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}

        # resize pos_embed if needed
        if "pos_embed" in state and hasattr(model, "pos_embed"):
            pos_embed_w = state["pos_embed"]
            if pos_embed_w.shape != model.pos_embed.shape:
                pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, "num_tokens", 1),
                                               model.patch_embed.grid_size)
                state["pos_embed"] = pos_embed_w

        msg = model.load_state_dict(state, strict=False)
        print(f"Pretrained weights found at {pretrained_weights} and loaded (msg={msg})")
        return

    # fallback: some presets by string key
    if pretrained_weights == "download":
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is not None:
            state = torch.hub.load_state_dict_from_url(url=f"https://dl.fbaipublicfiles.com/dino/{url}")
            pos_embed_w = state["pos_embed"]
            if pos_embed_w.shape != model.pos_embed.shape:
                pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, "num_tokens", 1),
                                               model.patch_embed.grid_size)
                state["pos_embed"] = pos_embed_w
            model.load_state_dict(state, strict=False)
            print(f"Downloaded and loaded weights from {url}")
            return

    if pretrained_weights == "supervised":
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "deit_small_patch16_224-cd65a155.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "deit_base_patch16_224-b5f2ef4d.pth"

        if url is not None:
            state_all = torch.hub.load_state_dict_from_url(url=f"https://dl.fbaipublicfiles.com/deit/{url}")
            state = state_all["model"]
            pos_embed_w = state["pos_embed"]
            if pos_embed_w.shape != model.pos_embed.shape:
                pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, "num_tokens", 1),
                                               model.patch_embed.grid_size)
                state["pos_embed"] = pos_embed_w
            msg = model.load_state_dict(state, strict=False)
            print(f"Loaded supervised weights ({url}), msg: {msg}")
            return

    print("No pretrained weights loaded; using random initialization.")


def load_mirage(model: torch.nn.Module, load_path: str) -> torch.nn.Module:
    """Adapt and load a MIRAGE checkpoint (special key naming conventions)."""
    state = torch.load(load_path, map_location="cpu")
    model_state = state.get("model", state)

    # rename keys to match current model
    for key in list(model_state.keys()):
        if key.startswith("encoder."):
            model_state[key.replace("encoder.", "blocks.")] = model_state.pop(key)

    # adjust some names used previously
    if "global_tokens" in model_state:
        model_state["cls_token"] = model_state.pop("global_tokens")

    # map some adapter names
    if "input_adapters.bscan.proj.weight" in model_state:
        model_state["patch_embed.proj.weight"] = model_state.pop("input_adapters.bscan.proj.weight")
        model_state["patch_embed.proj.bias"] = model_state.pop("input_adapters.bscan.proj.bias")

    # keep only keys present in model
    new_state = {}
    curr_keys = deepcopy(list(model_state.keys()))
    for k in curr_keys:
        if k in model.state_dict().keys():
            new_state[k] = model_state[k]

    model.load_state_dict(new_state, strict=False)
    print(f"MIRAGE weights loaded from {load_path}; loaded keys: {list(new_state.keys())}")
    return model


def load_model(load_path: str, model: torch.nn.Module,
               optimizer_net: Optional[torch.optim.Optimizer] = None,
               allow_size_mismatch: bool = False,
               device: str = "cuda:0"):
    """
    Load model checkpoint. Accepts multiple checkpoint key formats.
    If optimizer passed, also restore optimizer state and return (model, optimizer, epoch, checkpoint)
    """
    assert os.path.isfile(load_path), f"Checkpoint not found: {load_path}"
    if ("MIRAGE" in load_path) and allow_size_mismatch:
        return load_mirage(model, load_path)

    checkpoint = torch.load(load_path, map_location=device)
    if "model" in checkpoint:
        loaded_state = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        loaded_state = checkpoint["model_state_dict"]
    elif "teacher" in checkpoint:
        # special handler: use helper loader
        load_pretrained_weights_vfm(model, load_path, "teacher", "vit_base", 16)
        return model
    else:
        loaded_state = checkpoint

    if allow_size_mismatch:
        # remove mismatched keys
        model_state = model.state_dict()
        to_delete = []
        for k, v in list(loaded_state.items()):
            if k not in model_state or v.shape != model_state[k].shape:
                to_delete.append(k)
        for k in to_delete:
            del loaded_state[k]

    msg = model.load_state_dict(loaded_state, strict=not allow_size_mismatch)

    if optimizer_net is not None and "optimizer_state_dict" in checkpoint:
        optimizer_net.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer_net, checkpoint.get("epoch_idx", -1), checkpoint

    print(f"[INFO] Model loaded from {load_path}")
    print('[INFO] Loading message {}'.format(msg))
    return model


# ------------------------
# Visualization helpers
# ------------------------
def draw_results(results: Tuple[list, list, list, list], save_path: str) -> None:
    """
    Save loss and accuracy curves for train and val.
    results = [epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val]
    """
    os.makedirs(save_path, exist_ok=True)
    epochs_loss, epochs_acc, epochs_loss_val, epochs_acc_val = results

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_loss, label="train")
    plt.plot(epochs_loss_val, label="val")
    plt.title("Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_acc, label="train")
    plt.plot(epochs_acc_val, label="val")
    plt.title("Accuracy")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(save_path, "acc.png"))
    plt.close()


# ------------------------
# Convenience: JSON -> YAML for quick export
# ------------------------
def json2yaml(path: str) -> None:
    """Read {path}/config.json and dump {path}/output.yaml"""
    try:
        import yaml  # local import so not mandatory globally
    except ImportError:
        raise RuntimeError("Please install pyyaml to use json2yaml (pip install pyyaml)")

    src = os.path.join(path, "config.json")
    dst = os.path.join(path, "output.yaml")
    assert os.path.isfile(src), f"Missing JSON config: {src}"
    with open(src, "r") as fh:
        data = json.load(fh)
    with open(dst, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)
    print(f"Converted {src} -> {dst}")


def get_model_stats(model, A, B, device="cpu"):
    """
    Compute parameters and FLOPs for a PyTorch model.

    Args:
        model: nn.Module
        input_size: tuple, e.g. (1, 3, 224, 224)
        device: "cpu" or "cuda"

    Returns:
        dict containing:
            - params
            - params_m
            - flops
            - flops_g
    """
    model = model.to(device)
    A = A.to(device)
    B = B.to(device)
    model.eval()

    flops, params = profile(
        model.cuda(),
        inputs=(A.cuda(),B.cuda()),
        verbose=False
    )

    return {
        "params": params,
        "params_m": params / 1e6,
        "flops": flops,
        "flops_g": flops / 1e9,
    }
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Sampler
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale, ToPILImage, GaussianBlur
from torchvision.transforms.functional import InterpolationMode


class LowRes(nn.Module):
    """Randomly downscale and upscale the image to simulate low resolution."""

    def __init__(self, scale_range=(0.3, 0.7), p=0.5):
        super().__init__()
        self.scale_range = scale_range
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            c, h, w = img.shape
            img = F.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)
            img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        return img


class RandomBlur(nn.Module):
    """Apply Gaussian blur with random sigma."""

    def __init__(self, kernel_size=(3, 3), sigma_range=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            blur = T.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)
            img = blur(img)
        return img


class RandomGamma(nn.Module):
    """Random gamma adjustment."""

    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img = TF.adjust_gamma(img, gamma=gamma)
        return img


class RandomSharpen(nn.Module):
    """Random sharpness adjustment."""

    def __init__(self, sharpness_range=(0.5, 2.0), p=0.5):
        super().__init__()
        self.sharpness_range = sharpness_range
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.sharpness_range)
            img = TF.adjust_sharpness(img, sharpness_factor=factor)
        return img


# =====================================================
# 🟢 BalancedBatchSampler
# =====================================================
class BalancedBatchSampler(Sampler):
    """
    Samples batches with an equal number of samples per class.

    Ensures each batch contains the same number of samples for every class label.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.labels = dataset.labels
        self.shuffle = shuffle

        # Group sample indices by class label
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[int(label)].append(idx)

        self.labels_set = list(self.label_to_indices.keys())
        self.num_classes = len(self.labels_set)
        assert batch_size % self.num_classes == 0, \
            f"Batch size ({batch_size}) must be divisible by number of classes ({self.num_classes})."

        self.batch_size = batch_size
        self.samples_per_class = batch_size // self.num_classes
        self.num_batches = min(len(v) for v in self.label_to_indices.values()) // self.samples_per_class

    def __iter__(self):
        # Shuffle indices per class
        class_indices = {
            label: random.sample(indices, len(indices))
            for label, indices in self.label_to_indices.items()
        }

        # Yield balanced batches
        for batch_idx in range(self.num_batches):
            batch = []
            for label in self.labels_set:
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(class_indices[label][start:end])

            if self.shuffle:
                random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


# =====================================================
# 🟣 Transform Utilities
# =====================================================
def transform_custom(img_size=512, channels=1):
    """
    Create a torchvision transform for resizing, normalization, and grayscale conversion.

    Args:
        img_size (int): Target image size (square).
        channels (int): Number of output channels (1 = grayscale, 3 = RGB).

    Returns:
        torchvision.transforms.Compose:
    """
    mean, std = (
        [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]  # Note: based on ImageNet
        if channels > 1 else ([0.5], [0.5])
    )
    return Compose([
        ToPILImage(),
        Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        Grayscale(channels),
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])


def get_tensor(normalize=True, to_tensor=True):
    """Return a basic tensor transform with optional normalization."""
    transforms = []
    if to_tensor:
        transforms.append(ToTensor())
    if normalize:
        transforms.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return Compose(transforms)


def get_tensor_clip(normalize=True, to_tensor=True):
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    transforms = []
    if to_tensor:
        transforms.append(ToTensor())
    if normalize:
        transforms.append(Normalize(clip_mean, clip_std))
    return Compose(transforms)


# =====================================================
# 🟡 Image Utility Functions
# =====================================================

def denormalize_to_numpy(tensor: torch.Tensor):
    """
    Unnormalize a tensor image with arbitrary mean and std.

    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W) with values normalized.
        mean: list or tuple of means per channel
        std: list or tuple of stds per channel

    Returns:
        Tensor with values unnormalized.
    """

    mean, std = (torch.tensor((0.21091926, 0.21091926, 0.21091919)),
                 torch.tensor((0.17598894, 0.17598891, 0.17598893)))
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)  # Note channel first
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)  # Note channel first
    img = tensor * std + mean
    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


def bbox_process(bbox):
    """
    Convert [x, y, w, h] format to [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = map(int, bbox)
    return [x, y, x + w, y + h]

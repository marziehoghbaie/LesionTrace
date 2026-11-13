from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.utils import *

# --- Simplified Path Setup ---
add_project_paths(3)
from data import *
import random
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Sampler
from collections import defaultdict
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Grayscale, ToPILImage
from torchvision.transforms.functional import InterpolationMode

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


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
def transform_custom(img_size=512, channels=1, mask=False, train=False, CT=False):
    """
    Create a torchvision transform for resizing, normalization, and grayscale conversion.

    Args:
        img_size (int): Target image size (square).
        channels (int): Number of output channels (1 = grayscale, 3 = RGB).

    Returns:
        torchvision.transforms.Compose:
    """
    if mask:
        return Compose([Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR), ])
    if CT:
        mean, std = (
            [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]  # Note: based on ImgNet
            if channels > 1 else ([0.5], [0.5])
        )
    else:
        mean, std = (
            [(0.21091926, 0.21091926, 0.21091919), (0.17598894, 0.17598891, 0.17598893)]  # Note: based on VFM model
            if channels > 1 else ([0.5], [0.5])
        )
    if train:
        return Compose([
            ToPILImage(),
            Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            Grayscale(channels),
            ToTensor(),
            RandomBlur(kernel_size=(3, 3), sigma_range=(0.1, 1.5), p=0.5),
            RandomGamma(gamma_range=(0.7, 1.3), p=0.5),
            RandomSharpen(sharpness_range=(0.5, 2.0), p=0.5),
            Normalize(mean=mean, std=std),
        ])
    else:
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


# --- Unified Dataset Creation Helper ---
def _make_dataset(split, dataset_info, model_config, where, logger, transform, transform_mask, test_flag, img_size,
                  vis=False):
    """Helper to construct MARIO_Progression_OCT for a given split.
    :param transform_mask:
    """
    maxvit = True if model_config['model_type'] == 'MaxViT' else False
    dataset_name = dataset_info.get('dataset_name', 'MARIO')
    if dataset_name == 'MARIO':
        proto_path = os.path.join(where, dataset_info.get('proto_path')) if dataset_info.get('proto_path',
                                                                                             None) is not None else None
        transform_mask = transform_mask
        logger.info('[INFO] data is loaded for OCT...')
        csv_path = os.path.join(where, dataset_info[f'sampled_250_per_class_{split}_vis']) if vis else os.path.join(
            where, dataset_info[f'csv_{split}'])
        return MARIO_Progression_OCT(mario_csv_path=csv_path,
                                     base_path=os.path.join(where, dataset_info['mario_base_path']),
                                     subset=split,
                                     test=test_flag,
                                     logger=logger,
                                     transform=transform,
                                     num_classes=model_config['num_classes'], maxvit=maxvit)
    elif dataset_name == 'CSC':
        from data.CSC_dataset import CSC_OCT
        logger.info('[INFO] data is loaded for OCT...') if logger is not None else print(
            '[INFO] data is loaded for OCT...')

        return CSC_OCT(csv_path=dataset_info['csv_path'],
                       base_path=dataset_info['base_path'],
                       logger=logger,
                       transform=transform,
                       num_classes=model_config['num_classes'])
    elif dataset_name == 'MedSG':
        from data.CT_dataset import MedSG
        logger.info('[INFO] data is loaded for MedSG CT...')
        base_path = dataset_info['test_base_path'] if split == 'test' else dataset_info['base_path']

        return MedSG(json_path=os.path.join(where, dataset_info[f'json_{split}']),
                     base_path=os.path.join(where, base_path),  # note for test the base path is different
                     subset=split,
                     test=test_flag,
                     logger=logger,
                     num_classes=model_config['num_classes'],
                     transform=transform,
                     task=dataset_info.get('task', 'Registered_Diff'),
                     img_size=img_size)

    return None


# --- Dataloader Factory ---
def create_dataloaders(dataset_info, train_config, model_config, logger, where, vis=False, only_test=False):
    """
    Creates train/val/test dataloaders for MARIO OCT progression dataset.
    """
    # --- Build Transform ---
    transform_train = transform_custom(
        img_size=model_config['image_size'],
        channels=model_config['channels'], train=True
    )
    transform_test = transform_custom(
        img_size=model_config['image_size'],
        channels=model_config['channels'], train=False
    )
    transform_mask = transform_custom(
        img_size=model_config['image_size'],
        # img_size=14,
        mask=True
    )
    if only_test:
        # --- DataLoader Args ---
        common_args = dict(
            batch_size=train_config['batch_size'],
            num_workers=dataset_info.get('num_workers', 4),
            pin_memory=True,  # improves GPU data transfer speed
            persistent_workers=True  # reuse worker processes
        )

        test_set = _make_dataset('test', dataset_info, model_config, where, logger, transform_test, transform_mask,
                                 test_flag=True, img_size=model_config['image_size'], vis=vis)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common_args)
        return None, None, test_loader
    else:
        # --- Build Datasets ---
        train_set = _make_dataset('train', dataset_info, model_config, where, logger, transform_train, transform_mask,
                                  test_flag=False, img_size=model_config['image_size'], vis=vis)
        val_set = _make_dataset('val', dataset_info, model_config, where, logger, transform_test, transform_mask,
                                test_flag=True, img_size=model_config['image_size'], vis=vis)
        test_set = _make_dataset('test', dataset_info, model_config, where, logger, transform_test, transform_mask,
                                 test_flag=True, img_size=model_config['image_size'], vis=vis)

        # --- DataLoader Args ---
        common_args = dict(
            batch_size=train_config['batch_size'],
            num_workers=dataset_info.get('num_workers', 4),
            pin_memory=True,  # improves GPU data transfer speed
            persistent_workers=True  # reuse worker processes
        )

        # --- Train Loader ---
        if train_config.get('train', True):
            if model_config['model_type']=='Model_AmerensABekkers':
                logger.info('[INFO] data is loaded for Model_AmerensABekkers with batch sampling...')
                train_sampler = BalancedBatchSampler(train_set, train_config['batch_size'], shuffle=True)
                train_loader = DataLoader(train_set, batch_sampler=train_sampler)
            else:
                logger.info('[INFO] No Batch sampling ...')
                train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_args)
        else:
            train_loader = DataLoader(train_set, shuffle=False, drop_last=False, **common_args)

        # --- Val/Test Loaders ---
        val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common_args)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common_args)

        return train_loader, val_loader, test_loader

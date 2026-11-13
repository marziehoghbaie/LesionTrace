import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.utils import add_project_paths

add_project_paths(5)
from .channel_wise_aug import augmentations, tta_augs, available_augmentations
from collections import Counter


# --- Dataset Class ---
class MARIO_Progression_OCT(Dataset):
    def __len__(self):
        return len(self.data)

    def __init__(
            self,
            mario_csv_path,
            base_path,
            subset="train",
            test=False,
            logger=None,
            transform=None,
            transform_mask=None,
            img_size=512,
            visualize=False,
            proto_path=None,
            num_classes=3,
            include_offset_aug=False,
            test_aug=False):
        """
        Dataset for MARIO progression OCT scans.

        Args:
            mario_csv_path (str): Path to the CSV containing metadata.
            base_path (str): Root directory containing image folders.
            subset (str): Data subset ('train', 'val', 'test').
            test (bool): Disable augmentation if True.
            transform (callable): Image transformation pipeline.
            img_size (int): Image resize dimension.
            visualize (bool): Enable visualization mode.
        """
        self.logger = logger
        self.base_path = base_path
        self.subset = subset
        self.transform = transform
        self.transform_mask = transform_mask
        self.test_aug = test_aug
        assert self.transform is not None

        self.test = test
        self.visualize = visualize
        self.img_size = img_size
        # --- Load CSV ---
        self.data = pd.read_csv(mario_csv_path)
        # --- Define class mapping ---
        self.classes = {
            "Reduced": 0,
            "Stable": 1,
            "Increase": 2,
        }
        self.data = self.data[self.data['label'] != 'Uninterpretable'].reset_index(drop=True)

        # --- Preprocess file paths ---
        self.image_at_ti = self.data["image_at_ti"].apply(
            lambda x: os.path.join(self.base_path, subset, x)
        ).tolist()
        self.image_at_ti_1 = self.data["image_at_ti+1"].apply(
            lambda x: os.path.join(self.base_path, subset, x)
        ).tolist()

        # --- Labels and metadata ---
        self.labels = self.data["label"].map(self.classes)
        self.diffs = self.data["delta_t"]
        self.cases = self.data["case"]
        self.num_classes = num_classes

        # Optional metadata
        self.id_patient = self.data.get("id_patient", None)
        self.sex = self.data.get("sex", None)
        self.age_at_ti = self.data.get("age_at_ti", None)
        self.age_at_ti_1 = self.data.get("age_at_ti+1", None)

        # --- Augmentation ---
        self.augmentation_list = available_augmentations



        self.mask_valid_aug = ['rotate',
                               'eraser',
                               'flip_hr',
                               'offset_aug',
                               'random_crop_pair']

        if self.test_aug:
            self.augmentation_list = tta_augs
        elif not self.test_aug and self.test:
            self.augmentation_list = []
        if include_offset_aug:
            logger.info('[INFO] Offset augmentation is included in the augmentation list.')
            self.augmentation_list.append('offset_aug')
        logger.info(f'[INFO] Augmentation list for subset {subset}: {self.augmentation_list}')
        # read the prototypes
        self.proto_path = proto_path
        if self.proto_path:
            assert self.transform_mask is not None
            self.load_masks()
        self.mask_thresh = 0.1

        all_labels = self.data["label"]  # slow but only done once
        counts = Counter(all_labels)
        class_counts = torch.tensor([counts[i] for i in self.classes.keys()], dtype=torch.float)
        # Inverse frequency: rare classes get higher weight
        class_weights = (1.0 / class_counts)
        self.class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize to mean=1
        self.counts = counts
        logger.info(f'[INFO] Class weights: {self.class_weights}')
        logger.info(f'[INFO] Class weights: {counts}')


    def load_masks(self):
        self.sample_protoes = []
        _proto_paths_main = os.path.join(self.proto_path, self.subset)
        self.protoes = {}
        for sample in os.listdir(_proto_paths_main):
            splits = sample.split(sep='_')
            case_id = int(splits[1])
            img_name = str(splits[3])
            pred_label = str(splits[-1]).split(sep='.')[0]
            tmp_proto_path = os.path.join(_proto_paths_main, sample)
            self.protoes.update({(case_id, img_name): (tmp_proto_path, pred_label)})

        self.logger.info('[INFO] {} proto-masks have been loaded for subset {}'.format(len(self.protoes),
                                                                                       self.subset))

    def norm_map(self, map_inp):
        return (map_inp - map_inp.min()) / (map_inp.max() - map_inp.min() + 1e-8)

    def read_masks(self, cid, img0_name, img1_name, img0, img1):
        """
        Reads top-K proto-feature maps for two images, merges them,
        normalizes, and produces binary masks.
        """
        visits_binary_masks = []
        selected_proto_maps_ids_amd = [50, 640]
        for img_name, img in zip([img0_name, img1_name], [img0, img1]):
            # Load proto features
            proto_path, pred_label = self.protoes[(cid, img_name)]
            data = np.load(proto_path)
            blk_features = data['blk_features']
            similarities = data['similarities']
            (height, width) = img.shape[0], img.shape[1]
            if pred_label == 'AMD':
                top_maps = [self.norm_map(blk_features[fid]) for fid in selected_proto_maps_ids_amd]
                sim_top = [(similarities[fid]) for fid in selected_proto_maps_ids_amd]
                maps = np.stack(top_maps, axis=0)  # (K, H, W)
                weights = np.array(sim_top, dtype=np.float32)  # (K,)
                weights = weights / (weights.sum() + 1e-8)
                # Merge by taking maximum response across top-K maps
                # Merge by taking maximum response across top-K maps
                # merged = np.maximum.reduce(top_maps)
                merged = np.sum(maps * weights[:, None, None], axis=0)  # (H, W)

                # Normalize to [0, 1]
                merged = (merged - merged.min()) / (merged.max() - merged.min() + 1e-8)  # (14, 14)
                merged = cv2.resize(merged, (width, height))
                # Create binary mask
                binary_mask = (merged >= self.mask_thresh).astype(np.uint8)
                binary_mask = np.clip(binary_mask, 0, 1)

                assert binary_mask.max() <= 1 and binary_mask.min() >= 0, "Mask is not binary!"
            else:
                binary_mask = np.zeros((height, width), dtype=np.uint8)  # all ones for non-AMD cases
                # binary_mask = np.zeros((14, 14), dtype=np.uint8)  # all ones for non-AMD cases

            visits_binary_masks.append(binary_mask)  # Note (224, 224)

        return visits_binary_masks  # [mask_img0, mask_img1]

    def __getitem__(self, idx):
        # --- Load paired images ---
        img0_path = self.image_at_ti[idx]
        img1_path = self.image_at_ti_1[idx]
        cid = self.cases[idx]
        label = int(self.labels.iloc[idx])

        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        if img0 is None or img1 is None:
            raise FileNotFoundError(f"Missing image pair at index {idx}: {img0_path}, {img1_path}")

        # --- Initialize default masks (binary 0/1 float) ---
        (height0, width0) = img0.shape[0], img0.shape[1]
        (height1, width1) = img1.shape[0], img1.shape[1]

        maskA_binary = np.zeros((height0, width0), dtype=np.float32)  # shape: h, w, c
        maskB_binary = np.zeros((height1, width1), dtype=np.float32)

        # --- Load prototype-based masks if available ---
        if self.proto_path:  # not for uninterpretable
            visits_binary_masks = self.read_masks(cid,
                                                  os.path.splitext(os.path.basename(img0_path))[
                                                      0],
                                                  os.path.splitext(os.path.basename(img1_path))[
                                                      0],
                                                  img0, img1)

            maskA_binary, maskB_binary = visits_binary_masks

        # --- Apply augmentations if training ---
        if not self.test or self.test_aug:
            aug_name = random.choice(self.augmentation_list)
            if aug_name in self.mask_valid_aug:
                if aug_name == 'offset_aug':  # Note different offset for each image to compensate for misaligned images
                    img0, maskA_binary = augmentations[aug_name]([img0, maskA_binary])
                    img1, maskB_binary = augmentations[aug_name]([img1, maskB_binary])
                else:
                    img0, img1, maskA_binary, maskB_binary = augmentations[aug_name](
                        [img0, img1, maskA_binary, maskB_binary])
            else:
                img0 = augmentations[aug_name]([img0])[0]
                img1 = augmentations[aug_name]([img1])[0]


        # --- Convert masks to tensors, float32, keep shape (1, H, W) ---
        if len(maskA_binary.shape) == 3:
            # convert to gray scale
            maskA_binary = cv2.cvtColor(maskA_binary, cv2.COLOR_BGR2GRAY)
            maskB_binary = cv2.cvtColor(maskB_binary, cv2.COLOR_BGR2GRAY)

        maskA_binary = np.asarray(maskA_binary, dtype=np.float32)  # ensure float
        maskB_binary = np.asarray(maskB_binary, dtype=np.float32)

        # Add masks and clip to 0/1
        merged_mask = np.clip(maskA_binary + maskB_binary, 0, 1)  # shape: (H, W)

        maskA_binary = torch.tensor(maskA_binary).unsqueeze(0)
        maskB_binary = torch.tensor(maskB_binary).unsqueeze(0)
        merged_mask = torch.tensor(merged_mask).unsqueeze(0)

        # --- Apply transforms ---
        # Images: standard transforms
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        # Masks: add channel dim (1,H,W) and resize with nearest neighbor
        maskA_binary = (self.transform_mask(maskA_binary)).squeeze(0)  # (1,H_new,W_new)
        maskB_binary = (self.transform_mask(maskB_binary)).squeeze(0)
        merged_mask = (self.transform_mask(merged_mask)).squeeze(0)

        maskA_binary = (maskA_binary).float()
        maskB_binary = (maskB_binary).float()
        merged_mask = (merged_mask).float()

        maskA_binary = (maskA_binary > self.mask_thresh).float()  # (1,H_new,W_new)
        maskB_binary = (maskB_binary > self.mask_thresh).float()
        merged_mask = (merged_mask > self.mask_thresh).float()
        # --- Return sample dictionary ---

        return {
            "name": cid,
            "label": label,
            "path_A": img0_path,
            "path_B": img1_path,
            "A": img0,
            "B": img1,
            "maskA_binary": maskA_binary,
            "maskB_binary": maskB_binary,
            "merged_mask": merged_mask,
        }
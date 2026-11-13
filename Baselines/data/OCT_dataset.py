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
            num_classes=4,
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
            logger.info(f'[INFO] Augmentation list for subset {subset}: {self.augmentation_list}')

        if include_offset_aug:
            logger.info('[INFO] Offset augmentation is included in the augmentation list.')
            self.augmentation_list.append('offset_aug')
        # read the prototypes
        self.proto_path = proto_path
        if self.proto_path:
            assert self.transform_mask is not None
            self.load_masks()
        self.mask_thresh = 0.1

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



        # --- Apply augmentations if training ---
        if not self.test or self.test_aug:
            aug_name = random.choice(self.augmentation_list)
            if aug_name in self.mask_valid_aug:
                if aug_name == 'offset_aug':  # Note different offset for each image to compensate for misaligned images
                    img0 = augmentations[aug_name]([img0])
                    img1 = augmentations[aug_name]([img1])
                else:
                    img0, img1 = augmentations[aug_name]([img0, img1])
            else:
                img0 = augmentations[aug_name]([img0])[0]
                img1 = augmentations[aug_name]([img1])[0]




        # --- Apply transforms ---
        # Images: standard transforms
        img0 = self.transform(img0)
        img1 = self.transform(img1)


        return {
            "case_id": cid,
            "label": label,
            "path_A": img0_path,
            "path_B": img1_path,
            "A": img0,
            "B": img1,
        }

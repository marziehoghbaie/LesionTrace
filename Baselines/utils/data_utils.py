from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.utils import *

# --- Simplified Path Setup ---
add_project_paths(3)
from data import *


# --- Unified Dataset Creation Helper ---
def _make_dataset(split, dataset_info, model_config, where, logger, transform, test_flag, img_size,
                  vis=False, test_aug=False):
    """Helper to construct MARIO_Progression_OCT for a given split.
    """
    dataset_name = dataset_info.get('dataset_name', 'MARIO')
    if dataset_name == 'MARIO':
        logger.info('[INFO] data is loaded for OCT...')
        csv_path = os.path.join(where, dataset_info[f'sampled_250_per_class_{split}_vis']) if vis else os.path.join(
            where, dataset_info[f'csv_{split}'])
        return MARIO_Progression_OCT(mario_csv_path=csv_path,
                                     base_path=os.path.join(where, dataset_info['mario_base_path']),
                                     subset=split,
                                     test=test_flag,
                                     logger=logger,
                                     transform=transform,
                                     num_classes=model_config['num_classes'],
                                     test_aug=test_aug)
    elif dataset_name == 'CSC' or dataset_name == 'TREND_T_E' or dataset_name == 'TREND_Monthly':
        logger.info('[INFO] data is loaded for OCT...')
        return External_OCT(csv_path=dataset_info['csv_path'],
                            base_path=dataset_info['base_path'],
                            label_thresh=dataset_info['label_thresh'],
                            logger=logger,
                            transform=transform,
                            num_classes=model_config['num_classes'],
                            dataset_name=dataset_name, )

    return None


# --- Dataloader Factory ---
def create_dataloaders(dataset_info, train_config, model_config, logger, where, vis=False, only_test=False):
    """
    Creates train/val/test dataloaders for MARIO OCT progression dataset.
    """
    # --- Build Transform ---
    transform_train_test = transform_custom(img_size=model_config['image_size'],
                                            channels=model_config['channels'])

    if only_test or dataset_info['dataset_name'] == 'CSC' or dataset_info['dataset_name'] == 'TREND':
        # --- DataLoader Args ---
        common_args = dict(
            batch_size=train_config['batch_size'],
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )

        test_set = _make_dataset('test', dataset_info, model_config, where, logger, transform_train_test,
                                 test_flag=True, img_size=model_config['image_size'], vis=vis,
                                 test_aug=train_config.get('test_aug', False))
        test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common_args)
        return None, None, test_loader
    else:
        # --- Build Datasets ---
        train_set = _make_dataset('train', dataset_info, model_config, where, logger, transform_train_test,
                                  test_flag=False, img_size=model_config['image_size'], vis=vis,
                                  test_aug=train_config.get('test_aug', False))
        val_set = _make_dataset('val', dataset_info, model_config, where, logger, transform_train_test,
                                test_flag=True, img_size=model_config['image_size'], vis=vis,
                                test_aug=train_config.get('test_aug', False))
        test_set = _make_dataset('test', dataset_info, model_config, where, logger, transform_train_test,
                                 test_flag=True, img_size=model_config['image_size'], vis=vis,
                                 test_aug=train_config.get('test_aug', False))

        # --- DataLoader Args ---
        common_args = dict(
            batch_size=train_config['batch_size'],
            num_workers=dataset_info.get('num_workers', 4),
            pin_memory=True,  # improves GPU data transfer speed
            persistent_workers=True  # reuse worker processes
        )

        # --- Train Loader ---
        if train_config.get('train', True):
            if model_config['model_type'] == 'Model_AmerensABekkers':
                logger.info('[INFO] Batch Sampling is used...')
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

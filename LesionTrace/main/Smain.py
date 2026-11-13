import importlib
import os
import sys
import inspect
import warnings

import torch
import torch.nn as nn
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore", category=DeprecationWarning)

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
sys.path[:0] = [current_dir, parent_dir, grandparent_dir, great_grandparent_dir]

torch.multiprocessing.set_sharing_strategy('file_system')

from test_files.test_utils import *
from config import *


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def main(config_path: str, args):
    print('Torch available:', torch.cuda.is_available())

    (model_config, dataset_info, train_config, log_info, where,
     config, device, check_point_path, model_name, results_path, logger) = read_conf_file(config_path, args)
    logger.info(f'[INFO] Config loaded from {config_path}')
    logger.info(f'args: {args}')

    train_config['num_epochs'] = train_config.get('num_epochs', 0)
    logger.info(f'[INFO] # epochs: {train_config["num_epochs"]}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.vis:
        train_config['batch_size']=1
    only_test = dataset_info['dataset_name'] == 'CSC'
    logger.info(f'[INFO] Loading dataset ... only_test={only_test}')

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_info, train_config, model_config, logger, where,
        vis=args.vis, only_test=only_test
    )
    len_train_loader = len(train_loader) if train_config['train'] else 0

    if train_config['train']:
        logger.info(
            f"[INFO] Dataset: {dataset_info['dataset_name']} | "
            f"#train batches: {len(train_loader)} | #val batches: {len(val_loader)}"
        )
        logger.info(f"Bscan dataset categories: {train_loader.dataset.classes}")
    # ── Model ─────────────────────────────────────────────────────────────────
    model, module_name = create_model(model_config, logger, results_path)
    logger.info('Model created.')
    module = importlib.import_module(module_name)

    train_val = module.train_val
    build_optimizer_with_layer_decay = module.build_optimizer_with_layer_decay
    test = module.test

    # ── Loss function ─────────────────────────────────────────────────────────
    if model_config['loss'] == 'cross_entropy':
        # label_smoothing=0.1 prevents overconfident predictions on the
        # 3-class imbalanced problem (stable class typically dominates)
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=train_config.get('label_smoothing', 0.1)
        ).to(device)
    elif model_config['loss'] == 'focal':
        from functools import partial
        from utils.functional import focal_loss
        gamma = model_config.get('gamma', 2)
        logger.info('[INFO] gamma for focal loss is {}'.format(gamma))
        loss_fn = partial(focal_loss, gamma=gamma)
    elif model_config['loss'] == 'weightedFocal':
        class_weights = train_loader.dataset.class_weights
        class_weights = class_weights.clamp(min=0.5, max=1.5)
        logger.info(f'[INFO] Class weights: {class_weights}')
        loss_fn = FocalLoss(gamma=1.5, weight=class_weights.to(device)).to(device)
    elif model_config['loss'] == 'weightedCE':
        class_weights = train_loader.dataset.class_weights
        class_weights = class_weights.clamp(min=0.5, max=1.5)
        logger.info(f'[INFO] Class weights: {class_weights}')
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=train_config['label_smoothing']
        ).to(device)
    else:
        raise ValueError(f"Unknown loss: {model_config['loss']}")

    train_config['loss'] = model_config['loss']



    train_config['model_type'] = model_config['model_type']
    train_config['dataset_name'] = dataset_info['dataset_name']
    train_config['num_classes'] = model_config['num_classes']
    train_config['clamp'] = args.clamp
    logger.info('-' * 70)

    model = model.cuda()
    hash_before = get_model_hash(model)
    logger.info(f"Hash before loading: {hash_before}")
    init_epoch = 0
    train_config['base_lr'] = train_config.get('init_lr', 1e-4)
    use_ema = train_config.get('use_ema', True)

    if use_ema and not train_config['train']:  # only at the test time
        ema = module.ModelEMA(model, decay=train_config.get('ema_decay', 0.999))
        logger.info('[INFO] EMA enabled.')
    else:
        ema = None

    # Load pretrained / medical-imaging weights into backbone
    if train_config['pretrain'] or (
            not train_config['train'] and 'RetFiner_VisionFM.pth' not in check_point_path
    ):
        model = load_model(
            check_point_path, model,
            allow_size_mismatch=train_config['allow_size_mismatch'], ema=ema,
        )
        hash_after = get_model_hash(model)
        logger.info(f"[INFO] Checkpoint loaded from {check_point_path}")
        logger.info(f"Hash after loading: {hash_after}")
        assert hash_before != hash_after, \
            "Weights did not change after loading — check checkpoint path."

    # ── Optimizer ─────────────────────────────────────────────────────────────
    logger.info('-' * 70)
    freeze = train_config.get('freeze', False)
    if freeze:
        logger.info('[INFO] v2 optimizer: layer-wise LR decay for backbone, full LR for new modules.')
        optimizer = build_optimizer_with_layer_decay(
            model=model,
            base_lr=train_config.get('init_lr', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            layer_decay=train_config.get('layer_decay', 0.75),
        )
    else:
        # In build_optimizer_with_layer_decay, or directly in main.py:
        # Backbone gets ~50-100x lower LR than new modules
        # This achieves the same protection as freezing without creating
        # a two-phase optimization landscape that encourages overfitting
        optimizer = build_optimizer_with_layer_decay(
            model=model,
            base_lr=train_config.get('init_lr', 1e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            layer_decay=train_config.get('layer_decay', 0.55),  # aggressive — block 0 gets ~1e-4 * 0.55^12 ≈ 8e-7
        )
    scheduler = create_scheduler(train_config, optimizer, len_train_loader)
    ckpt_saver = CheckpointSaver(results_path)
    logger.info(f"[INFO] Training config:\n{config}")
    train_config['num_classes'] = model_config['num_classes']

    # ── Resume ────────────────────────────────────────────────────────────────
    if train_config['resume']:
        model, optimizer, init_epoch, ckpt_dict = load_model(
            check_point_path, model, optimizer,
            allow_size_mismatch=train_config['allow_size_mismatch']
        )
        ckpt_saver.set_best_from_ckpt(ckpt_dict)
        logger.info(f"[INFO] Resuming from {check_point_path}")
        init_epoch = 0  # set to 0 for fine-tuning; remove this line to truly resume
        logger.info(f"[INFO] Init epoch set to {init_epoch}")

    # ── Train or evaluate ─────────────────────────────────────────────────────
    if train_config['train']:
        # train_val is now always imported from vit_change_detection_v2.
        # The args.no_stable flag controlled which test function was used
        # inside the loop — this is handled inside train_val itself via
        # the test import from test_files.test_no_stable.
        logger.info('[INFO] Starting training with train_val loop.')
        if freeze:
            logger.info(f'[INFO] Two-stage training: backbone frozen for '
                        f'{train_config.get("freeze_backbone_epochs", 10)} epochs.')

        train_val(
            train_config, logger, model, optimizer, loss_fn,
            train_loader, val_loader, scheduler, ckpt_saver,
            init_epoch, args
        )
        logger.info('[INFO] End of training phase.')

    else:
        # ── Evaluation ────────────────────────────────────────────────────────
        model.eval()

        if args.test_with_validation:
            dataset_info['test_with_validation'] = True
            current_set = "val"
        else:
            dataset_info['test_with_validation'] = False
            current_set = "test"

        # CSC dataset always uses the test set
        if dataset_info['dataset_name'] == 'CSC':
            dataset_info['test_with_validation'] = False
            current_set = "test"

        loader = val_loader if dataset_info['test_with_validation'] else test_loader
        logger.info(f'[INFO] Evaluating on {current_set} set '
                    f'({len(loader.dataset)} samples).')

        if args.vis:
            from test_files.vis import vis_maps
            train_config['dataset_name'] = dataset_info['dataset_name']
            train_config['subset'] = current_set
            vis_maps(model, loader, logger, results_path)

        else:
            model.eval()
            if dataset_info['dataset_name'] == 'CSC' or dataset_info['dataset_name'] == 'TREND_T_E' or dataset_info['dataset_name'] == 'TREND_Monthly':
                module.test_external(loader, model, logger, save_path=results_path)
            else:
                test(loader, model, loss_fn, logger,
                     complete=True, save_path=results_path, train_config=train_config)
                change_bootstrap_CI(results_path)
                tsne_plot(t_npy_path=results_path)
                plot_umap(results_path, n_neighbors=15, min_dist=0.1,
                          n_components=3, metric='cosine', title='UMAP Projection')


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args.config_path, args)

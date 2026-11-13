import os
import sys
import inspect
import warnings

import torch
import torch.nn as nn

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Path setup for local imports
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
great_great_grandparent_dir = os.path.dirname(great_grandparent_dir)

sys.path[:0] = [current_dir, parent_dir, grandparent_dir, great_grandparent_dir, great_great_grandparent_dir]

# Multiprocessing strategy for torch
torch.multiprocessing.set_sharing_strategy('file_system')

# Local imports
from utils import *
from test_files import *
from config import *


# ------------------------------
# Main function
# ------------------------------
def main(config_path: str, args):
    print('Torch available:', torch.cuda.is_available())

    # Load configurations
    (model_config, dataset_info, train_config, log_info, where,
     config, device, check_point_path, model_name, results_path, logger) = read_conf_file(config_path)
    # Create model
    model = create_model(model_config, logger, device=device, check_point_path=check_point_path)
    logger.info('Model created.')
    logger.info('-' * 70)
    dataset_info['label_thresh'] = 150 # best selected one
    # Create results folder
    os.makedirs(results_path, exist_ok=True)

    # Load datasets
    logger.info('[INFO] Loading dataset ...')

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_info, train_config, model_config, logger, where,
        only_test=not train_config['train'] and not args.subset == 'val'
    )

    len_train_loader = len(train_loader) if train_config['train'] else 0
    if train_config['train']:
        logger.info(
            f"[INFO] Dataset: {dataset_info['dataset_name']} | "
            f"#train: {len(train_loader)} | #val: {len(val_loader)}"
        )
        logger.info(f"Bscan dataset categories: {train_loader.dataset.classes}")

    #  Set up the loss function
    model_type = model_config['model_type']
    if train_config['train']:
        weights = train_loader.dataset.class_weights if hasattr(train_loader.dataset, 'class_weights') else None
        if model_type in ['Model_YidingHao', 'Model_LovreAntonioBudimir']:  # Lumine, Model_LovreAntonioBudimir
            loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
        elif model_type in ['Model_AlbertoJBeltranCarrero']:
            loss_fn = focal_loss
        else:  # Model_PhilippeZhang, 'SimaesRetfound', 'SWINBiomarker', Model_AmerensABekkers (TONIC), Huber
            loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)
    #

    train_config["model_type"] = model_config["model_type"]
    # Optimizer & scheduler
    optimizer = create_optimizer(train_config, model)
    scheduler = create_scheduler(train_config, optimizer, len_train_loader)
    ckpt_saver = CheckpointSaver(results_path)

    model = model.cuda()
    hash_before = get_model_hash(model)
    logger.info(f"Hash before loading: {hash_before}")
    init_epoch = 0
    # Load checkpoint if needed
    if train_config['resume']:
        model, optimizer, init_epoch, ckpt_dict = load_model(
            check_point_path, model, optimizer, allow_size_mismatch=train_config['allow_size_mismatch']
        )
        ckpt_saver.set_best_from_ckpt(ckpt_dict)
        logger.info(f"[INFO] Resuming training from {check_point_path}")
        init_epoch = 0  # debug only for finetuning
        logger.info(f"[INFO] Init epoch is set to {init_epoch}")

    elif train_config['pretrain'] or not train_config['train']:
        if 'RETFound_oct_weights.pth' not in check_point_path:
            model = load_model(check_point_path, model, allow_size_mismatch=train_config['allow_size_mismatch'])
            hash_after = get_model_hash(model)
            logger.info(f"[INFO] Checkpoint loaded from {check_point_path}")
            logger.info(f"Hash after loading: {hash_after}")
            assert hash_before != hash_after

    logger.info(f"[INFO] Training config:\n{config}")

    # Training or evaluation
    if train_config['train']:
        from train_files.train import train_val
        assert train_config['num_epochs'] == 50
        train_val(
            train_config, logger, model, optimizer, loss_fn,
            train_loader, val_loader, scheduler, ckpt_saver,
            init_epoch, late_fusion=model_config['late_fusion']
        )
        logger.info('[INFO] End of training phase.')
    else:
        model.eval()
        assert not train_config['allow_size_mismatch']
        loader = val_loader if args.subset == 'val' else test_loader

        logger.info(f"[INFO] Tests on {args.subset} set...")
        train_config['subset'] = args.subset
        train_config['dataset_name'] = dataset_info['dataset_name']
        logger.info(
            f"[INFO] Dataset: {dataset_info['dataset_name']} | "
            f"#loader: {len(loader)}"
        )
        try:
            logger.info(f"Bscan dataset categories: {loader.dataset.classes}")
        except:
            logger.info(f"No labels for this dataset ...")
        if args.vis:
            vis_type = model_config.get('vis_type', 'gradcam')
            logger.info(f"[INFO] visualization Type :{vis_type}")
            logger.info('[INFO] Running Grad-CAM visualization ...')
            GradCAMVis(model, loader, logger, train_config, results_path, model_config)
        elif args.fluid_calc:
            logger.info('[INFO] Running Fluid calc visualization ...')
            from test_files.Fluid_change_detection import Fluid_change_prediction_GradCAM
            logger.info('Separate gradcam for each visit')
            Fluid_change_prediction_GradCAM(model, loader, logger, train_config, results_path,
                                            model_config)
        else:
            model.eval()
            test(loader, model, loss_fn, logger, late_fusion=model_config['late_fusion'], complete=True,
                 save_path=results_path, train_config=train_config)
            # if dataset_info['dataset_name'] == 'CSC':
            #     external_test = True
            # else:
            #     external_test = False
            # change_bootstrap_CI(results_path, external_test=external_test)
            # tsne_plot(t_npy_path=results_path)
            # plot_umap(results_path, n_neighbors=15, min_dist=0.1, n_components=3, metric='cosine',
            #           title='UMAP Projection')


# ------------------------------
# Entry point
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT2D', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args.config_path, args)

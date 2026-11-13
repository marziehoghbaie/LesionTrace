import inspect
import json
import os
import sys
import time
import shutil
import torch
import yaml
from utils import *

# ------------------------------
# Path setup for local imports
# ------------------------------
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for _ in range(3):
    sys.path.insert(0, current_dir)
    current_dir = os.path.dirname(current_dir)


def read_conf_file(config_path, args=None):
    """
    Reads a YAML configuration file and sets up:
      - model/dataset/train/log configs
      - paths (checkpoint, results)
      - logger and device
      - reproducibility setup (saves configs and relevant scripts)
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_config = config[0]["model_inputs"]
    dataset_info = config[1]["dataset"]
    train_config = config[2]['train_config']
    log_info = config[3]['log']
    where = config[4]['where']
    if args is not None and args.top_k:
        model_config['top_k'] = args.top_k
    else:
        model_config['top_k'] = model_config.get('top_k', 5)

    # ------------------------------
    # Merge relevant info into train_config
    # ------------------------------
    model_type = model_config.get('model_type', '')
    train_config.update({
        'loss': model_config.get('loss'),
        'save_metric': model_config.get('save_metric'),
        'siamese': 'Siamese' in model_type,
        'model_type': model_type,
        'GradCAM': train_config.get('GradCAM', False),
        'linear_probing': model_config.get('linear_probing', False),
        'cls_token': model_config.get('cls_token')
    })

    # ------------------------------
    # Device setup
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() and train_config.get('use_gpu', True) else 'cpu')

    # ------------------------------
    # Path setup
    # ------------------------------
    check_point_path = os.path.join(where, train_config['load_path'], train_config['checkpoint'])
    model_name = f"{model_type}_{model_config['image_size']}_{dataset_info['dataset_name']}"
    weight_postfix = os.path.splitext(os.path.basename(config_path))[0].split('_')[-1]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tta_prefix = 'tta' if train_config.get('test_aug', False) else 'no_tta'
    if not train_config['train']:
        log_info['save_path'] = os.path.join(where, train_config['load_path'])
        results_path = os.path.join(log_info['save_path'], f'Test_{tta_prefix}', dataset_info['dataset_name'], timestamp)
    else:
        results_path = os.path.join(where, log_info['save_path'], weight_postfix, model_name, timestamp)

    os.makedirs(results_path, exist_ok=True)

    # ------------------------------
    # Logger setup
    # ------------------------------
    logger = create_logger(results_path)
    logger.info(f"[INFO] Device: {device}")
    logger.info(f"[INFO] Results directory: {results_path}")
    logger.info('[INFO] the top-k is set to {}...'.format(model_config['top_k']))

    # ------------------------------
    # Save configs
    # ------------------------------
    with open(os.path.join(results_path, f"{model_name}.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    experiment_type = 'linear_probing' if model_config.get('linear_probing') else 'train'
    config_name = f"{experiment_type}_{model_type}_{model_config['image_size']}_save_metric-{model_config['save_metric']}_loss_{model_config['loss']}"
    with open(os.path.join(results_path, f"{config_name}.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return (
        model_config, dataset_info, train_config, log_info, where,
        config, device, check_point_path, model_name, results_path, logger
    )

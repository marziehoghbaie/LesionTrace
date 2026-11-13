import functools
import importlib
import inspect
import os
import shutil
import sys

import torch

# ---------------------------------
# Path setup for local imports
# ---------------------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for _ in range(4):
    sys.path.insert(0, currentdir)
    currentdir = os.path.dirname(currentdir)

torch.multiprocessing.set_sharing_strategy('file_system')


# ---------------------------------
# Helper functions
# ---------------------------------
def _import_and_copy_module(module_name: str, results_path: str, logger):
    """Import a module dynamically and copy its source file to results_path."""
    model_module = importlib.import_module(module_name)
    model_file = os.path.abspath(model_module.__file__)
    logger.info(f"[INFO] {module_name} is loaded from {model_file}")

    # Copy file for reproducibility
    dest_file = os.path.join(results_path, os.path.basename(model_file))
    try:
        shutil.copy(model_file, dest_file)
        logger.info(f"[INFO] Saved a copy of {model_file} to {dest_file}")
    except Exception as e:
        logger.warning(f"[WARNING] Could not copy module file: {e}")

    return model_module


def freeze_backbone(model, logger):
    logger.info("[INFO] Freezing model backbone ...")

    for name, p in model.named_parameters():
        if "head" in name:  # keep the head trainable
            p.requires_grad = True
        else:
            p.requires_grad = False


def unfreeze_all(model, logger):
    for p in model.parameters():
        p.requires_grad = True
    logger.info("[INFO] Unfreezing model backbone ...")


def _freeze_model(model, logger):
    """Freeze model parameters for linear probing."""
    logger.info("[INFO] Freezing model backbone for linear probing...")
    for param in model.parameters():
        param.requires_grad = False


# ---------------------------------
# Main model factory
# ---------------------------------

@functools.lru_cache(None)
def cached_import(module_name, results_path, logger):
    return _import_and_copy_module(module_name, results_path, logger)


def create_model(model_config, logger, results_path):
    model_type = model_config.get("model_type")
    img_size = model_config["image_size"]
    num_classes = model_config["num_classes"]
    in_chans = model_config["channels"]
    # --- Model module selection --- #
    ablation_type = model_config.get('ablation_type', None)
    module_map = {
        # "ChangeDetector": ("model_files.models", {'ablation': ablation_type}),
        "LesionChangeDetector": ("model_files.models", {}),
        "modelsNodelta": ("model_files.modelsNodelta", {}),
        "modelsNoSpatial": ("model_files.modelsNoSpatial", {}),
        "modelsOnlyInsideandDelta": ("model_files.modelsOnlyInsideandDelta", {}),
        "modelsOnlyOutsideandDelta": ("model_files.modelsOnlyOutsideandDelta", {}),
        # "LesionChangeDetectorImp": ("model_files.modelsImp", {}),
        # "LesionChangeDetectorImp2": ("model_files.modelsImp2", {}),
        # "models_simple": ("model_files.models_simple", {}),
        # "models_ExtraToken": ("model_files.models_ExtraToken", {}),
        # "modelsExtratokensV2": ("model_files.modelsExtratokensV2", {}),
    }
    module_name, kwargs = module_map.get(model_type)

    logger.info(f"[INFO] Creating model from {module_name}...")
    model_module = cached_import(module_name, results_path, logger)

    # --- Model instantiation ---
    model_backbone = model_config["back_bone"]
    kwargs['img_size'] = img_size
    kwargs['in_chans'] = in_chans
    model_class = getattr(model_module, model_backbone)

    model = model_class(
        pretrained=model_config['imgnet_pretrained'],
        num_classes= num_classes,
        **kwargs
    )
    return model, module_name

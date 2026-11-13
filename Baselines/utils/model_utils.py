import inspect
import os
import sys
import timm
import torch

# --- Path setup ---
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.extend([
    currentdir,
    os.path.dirname(currentdir),
    os.path.dirname(os.path.dirname(currentdir)),
    os.path.dirname(os.path.dirname(os.path.dirname(currentdir))),
])
torch.multiprocessing.set_sharing_strategy('file_system')


def create_model(model_config, logger, device='cpu', check_point_path=None):
    """
    Factory function to create a model based on the specified configuration.
    Supports custom models and TIMM-based architectures.
    """

    model_type = model_config.get('model_type')
    num_classes = model_config.get('num_classes', 4)
    in_chans = model_config.get('channels', 3)
    pretrained = model_config.get('imgnet_pretrained', True)
    img_size = model_config.get('image_size', 224)

    logger.info(f"[INFO] Initializing model: {model_type}")

    # --- Registry for baseline models ---
    baseline_models = {
        'Model_AlbertoJBeltranCarrero': lambda: __import__(
            'models.baselines.Model_AlbertoJBeltranCarrero', fromlist=['Model_AlbertoJBeltranCarrero']
        ).Model_AlbertoJBeltranCarrero(
            in_channels=in_chans, num_classes=num_classes,
            img_size=img_size, backbone=model_config.get('backbone', 'retfound'), device=device,
            backbone_chkp=check_point_path  # initialized by retfound
        ),
        'Model_AmerensABekkers': lambda: __import__(
            'models.baselines.Model_AmerensABekkers', fromlist=['Model_AmerensABekkers']
        ).Model_AmerensABekkers(num_classes=num_classes),  # loaded from IMGNet
        'Model_LovreAntonioBudimir': lambda: __import__(
            'models.baselines.Model_LovreAntonioBudimir', fromlist=['Model_LovreAntonioBudimir']
        ).Model_LovreAntonioBudimir(num_classes=num_classes, check_point_path=check_point_path,
                                    backbone=model_config.get('backbone', 'vit_large')),  # initialized by retfound
        'Model_MarcelHuber': lambda: __import__(
            'models.baselines.Model_MarcelHuber', fromlist=['Model_MarcelHuber']
        ).Model_MarcelHuber(num_classes=num_classes, pretrained=True),  # dino weights are initialized
        'Model_PhilippeZhang': lambda: __import__(
            'models.baselines.Model_PhilippeZhang', fromlist=['Model_PhilippeZhang']
        ).Model_PhilippeZhang(num_classes=num_classes, pretrained=True),  # resnet50 weights based on IMGNet
        'Model_YiDing': lambda: __import__(
            'models.baselines.Model_YiDing', fromlist=['Model_YiDing']
        ).Model_YiDing(num_classes=num_classes, pretrained=True),  # convnext large v2 weights based on IMGNet
        'Model_YidingHao': lambda: __import__(
            'models.baselines.Model_YidingHao', fromlist=['Model_YidingHao']
        ).Model_YidingHao(num_classes=num_classes, pretrained=True),  # convnext_large weights based on IMGNet
        'Model_YidingHao4Vis': lambda: __import__(
            'models.baselines.Model_YidingHao', fromlist=['Model_YidingHao4Vis']
        ).Model_YidingHao4Vis(num_classes=num_classes, pretrained=True),  # convnext_large weights based on IMGNet
        'Model_YidingHaoNoAttn': lambda: __import__(
            'models.baselines.Model_YidingHao', fromlist=['Model_YidingHaoNoAttn']
        ).Model_YidingHaoNoAttn(num_classes=num_classes, pretrained=True),  # convnext_large weights based on IMGNet
        'MaxViT': lambda: __import__(
            'models.baselines.MaxViT', fromlist=['MaxViT']
        ).MaxViT(num_classes=num_classes, pretrained=True),  # 'maxvit_tiny_tf_512.in1k' weights based on IMGNet
    }

    # --- Model Selection ---
    if model_type == 'SWINBiomarker':
        from models.SWIN_BioMarker.swin_late import swin_base_patch4_window7_224
        model = swin_base_patch4_window7_224(
            num_classes=num_classes, pretrained=pretrained, in_chans=in_chans
        )
        logger.info("[INFO] Late fusion mode (SWINBiomarker)")
    elif model_type == 'SimaesRetfound':
        from models.SimaesRetfound.models import SiameseNetwork
        model = SiameseNetwork(
            nb_classes=num_classes, device=device, check_point_path=check_point_path,
            backbone=model_config.get('backbone', 'vit_large')
        )
        logger.info("[INFO] Late fusion mode (SimaesRetfound)")
    elif model_type == 'Early_Siamese_RetFound':
        from models.baselines.Early_Siamese_RetFound import EarlySiameseNetwork
        model = EarlySiameseNetwork(nb_classes=num_classes, device=device, check_point_path=check_point_path, backbone=model_config.get('backbone', 'vit_large'))
        logger.info("[INFO] Early fusion mode (SimaesRetfound)")
    elif model_type in baseline_models:
        model = baseline_models[model_type]()
    else:
        # Fallback to timm models
        model = timm.create_model(
            model_type, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans
        )
        logger.info(f"[INFO] Created TIMM model: {model_type}")


    # Freezing parts according to the papers
    if model_type == 'Model_AlbertoJBeltranCarrero':
        # freeze the patch embedding layer (model.net.patch_embed)
        for param in model.net.patch_embed.parameters():
            param.requires_grad = False
        logger.info (f"[INFO] Freezing patch embedding layer for {model_type}.")

    logger.info(f"[INFO] Model {model_type} successfully initialized.")
    return model

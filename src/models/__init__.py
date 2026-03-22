"""
Model builders for semantic segmentation.
Uses segmentation-models-pytorch (smp) for quick prototyping.
"""

from .unet import UNet
from .hrnet import HRNet
from .pointflow import PointFlowNet

import segmentation_models_pytorch as smp


def build_unet(encoder="resnet34", encoder_weights="imagenet", num_classes=11, **kwargs):
    """Build U-Net model."""
    return UNet(encoder=encoder, encoder_weights=encoder_weights, num_classes=num_classes)


def build_unetpp(encoder="resnet34", encoder_weights="imagenet", num_classes=11, **kwargs):
    """Build U-Net++ model."""
    return smp.UnetPlusPlus(
        encoder_name=encoder, encoder_weights=encoder_weights,
        in_channels=3, classes=num_classes, activation=None,
    )


def build_deeplabv3plus(encoder="resnet50", encoder_weights="imagenet", num_classes=11, **kwargs):
    """Build DeepLabV3+ model."""
    return smp.DeepLabV3Plus(
        encoder_name=encoder, encoder_weights=encoder_weights,
        in_channels=3, classes=num_classes, activation=None,
    )


def build_hrnet(encoder="tu-hrnet_w18", encoder_weights="imagenet", num_classes=11, **kwargs):
    """Build HRNet model."""
    variant = encoder.replace("tu-hrnet_", "") if encoder.startswith("tu-hrnet_") else "w18"
    return HRNet(variant=variant, encoder_weights=encoder_weights, num_classes=num_classes)


def build_pointflow(encoder="resnet50", encoder_weights="imagenet", num_classes=11, **kwargs):
    """Build PointFlow model."""
    return PointFlowNet(encoder=encoder, encoder_weights=encoder_weights, num_classes=num_classes)


MODEL_REGISTRY = {
    "unet": build_unet,
    "unetpp": build_unetpp,
    "deeplabv3plus": build_deeplabv3plus,
    "hrnet": build_hrnet,
    "pointflow": build_pointflow,
}


def build_model(name: str, **kwargs):
    """Build a model by name from registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

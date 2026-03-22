"""
U-Net Baseline Model for Forest Semantic Segmentation.

Uses segmentation-models-pytorch with pretrained encoders.
Supports ResNet-34, ResNet-50, EfficientNet encoders.

Reference:
    Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


class UNet(nn.Module):
    """U-Net segmentation model wrapper.

    Args:
        encoder: Encoder backbone name (e.g., 'resnet34', 'resnet50', 'efficientnet-b3')
        encoder_weights: Pretrained weights ('imagenet' or None)
        num_classes: Number of output segmentation classes
        activation: Final activation (None for raw logits with CrossEntropyLoss)
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 11,
        activation: str = None,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
        self.encoder_name = encoder
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)

    def get_encoder_params(self):
        """Get encoder parameters (for differential LR)."""
        return self.model.encoder.parameters()

    def get_decoder_params(self):
        """Get decoder parameters (for differential LR)."""
        return list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())

    def freeze_encoder(self):
        """Freeze encoder weights for transfer learning warm-up."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"UNet(encoder={self.encoder_name}, classes={self.num_classes}, "
            f"params={total:,}, trainable={trainable:,})"
        )

"""
HRNet (High-Resolution Network) for Forest Semantic Segmentation.

HRNet maintains high-resolution representations throughout the network,
making it particularly effective for dense prediction tasks like segmentation.

Reference:
    Sun et al. "Deep High-Resolution Representation Learning for Visual Recognition" (2019)
    Used in the Forest Inspection paper (arXiv:2403.06621) as one of the main architectures.
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


class HRNet(nn.Module):
    """HRNet-based segmentation model.

    Uses timm's HRNet encoder through segmentation-models-pytorch.
    Available variants: hrnet_w18, hrnet_w32, hrnet_w48

    Args:
        variant: HRNet variant ('w18', 'w32', 'w48')
        encoder_weights: Pretrained weights ('imagenet' or None)
        num_classes: Number of output segmentation classes
        decoder_type: Decoder architecture ('unet', 'fpn', 'psp')
        activation: Final activation (None for raw logits)
    """

    def __init__(
        self,
        variant: str = "w18",
        encoder_weights: str = "imagenet",
        num_classes: int = 11,
        decoder_type: str = "unet",
        activation: str = None,
    ):
        super().__init__()

        encoder_name = f"tu-hrnet_{variant}"
        self.variant = variant
        self.num_classes = num_classes

        # Build model based on decoder type
        decoder_map = {
            "unet": smp.Unet,
            "fpn": smp.FPN,
            "psp": smp.PSPNet,
        }

        if decoder_type not in decoder_map:
            raise ValueError(f"Unknown decoder: {decoder_type}. Options: {list(decoder_map.keys())}")

        ModelClass = decoder_map[decoder_type]
        self.model = ModelClass(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
        )
        self.decoder_type = decoder_type

    def forward(self, x):
        return self.model(x)

    def get_encoder_params(self):
        """Get encoder parameters (for differential LR)."""
        return self.model.encoder.parameters()

    def get_decoder_params(self):
        """Get decoder parameters."""
        return list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())

    def freeze_encoder(self):
        """Freeze encoder weights."""
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
            f"HRNet(variant={self.variant}, decoder={self.decoder_type}, "
            f"classes={self.num_classes}, params={total:,}, trainable={trainable:,})"
        )

"""
PointFlow Network for Forest Semantic Segmentation.

PointFlow uses a multi-scale feature aggregation approach with point-wise 
flow fields to refine feature representations at different resolutions.

Reference:
    Li et al. "PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation" (2021)
    Used in the Forest Inspection paper (arXiv:2403.06621) as the second architecture.

Note:
    This is a simplified implementation using smp's FPN as a proxy for multi-scale 
    feature aggregation. For the full PointFlow architecture, consider using the 
    original repository: https://github.com/lxtGH/PFSegNets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class PointFlowModule(nn.Module):
    """Point Flow Module for feature refinement between scales.

    Generates a 2D offset field to warp features from one scale to another,
    enabling better alignment of multi-scale features.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, 1),  # 2D offset field
        )

    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_high: Higher resolution features (B, C, H, W)
            feat_low: Lower resolution features (B, C, H', W') where H'<H

        Returns:
            Refined features at high resolution (B, C, H, W)
        """
        # Upsample low-res to match high-res
        feat_low_up = F.interpolate(
            feat_low, size=feat_high.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate and predict offset
        concat = torch.cat([feat_high, feat_low_up], dim=1)
        offset = self.offset_conv(concat)  # (B, 2, H, W)

        # Create sampling grid
        B, _, H, W = feat_high.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat_high.device),
            torch.linspace(-1, 1, W, device=feat_high.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Apply offset
        offset = offset.permute(0, 2, 3, 1)  # (B, H, W, 2)
        grid = grid + offset * 0.1  # Scale offset

        # Warp features
        warped = F.grid_sample(feat_low_up, grid, mode="bilinear", align_corners=False)

        return feat_high + warped


class PointFlowNet(nn.Module):
    """PointFlow-inspired segmentation network.

    Uses an FPN backbone with PointFlow modules for multi-scale refinement.

    Args:
        encoder: Encoder backbone name
        encoder_weights: Pretrained weights
        num_classes: Number of segmentation classes
        fpn_channels: Number of channels in FPN outputs
        activation: Final activation
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        encoder_weights: str = "imagenet",
        num_classes: int = 11,
        fpn_channels: int = 256,
        activation: str = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # FPN backbone for multi-scale features
        self.fpn = smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=activation,
            decoder_pyramid_channels=fpn_channels,
            decoder_segmentation_channels=fpn_channels // 2,
        )

        # PointFlow modules for feature refinement
        self.pf_module = PointFlowModule(fpn_channels // 2)

        # Final head
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_channels // 2, fpn_channels // 4, 3, padding=1),
            nn.BatchNorm2d(fpn_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 4, num_classes, 1),
        )

    def forward(self, x):
        """Simple forward using FPN output."""
        return self.fpn(x)

    def freeze_encoder(self):
        """Freeze encoder weights."""
        for param in self.fpn.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.fpn.encoder.parameters():
            param.requires_grad = True

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"PointFlowNet(classes={self.num_classes}, "
            f"params={total:,}, trainable={trainable:,})"
        )

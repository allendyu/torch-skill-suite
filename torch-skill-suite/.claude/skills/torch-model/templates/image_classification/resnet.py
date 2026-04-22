"""ResNet model template for image classification.

Supports resnet18, resnet34, resnet50.
Uses torchvision's ResNet implementation with configurable head.
"""

import torch.nn as nn
from torchvision import models


RESNET_VARIANTS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

RESNET_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


def build_resnet(config: dict) -> nn.Module:
    """Build a ResNet model from configuration.

    Args:
        config: Dictionary with keys:
            - backbone: One of resnet18, resnet34, resnet50.
            - pretrained: Whether to load ImageNet pretrained weights.
            - in_channels: Number of input channels (3 for RGB).
            - head: Dict with head configuration (type, num_classes, pooling, dropout).

    Returns:
        nn.Module model instance (backbone + head).
    """
    backbone_name = config.get("backbone", "resnet34")
    if backbone_name not in RESNET_VARIANTS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {sorted(RESNET_VARIANTS.keys())}")

    pretrained = config.get("pretrained", True)
    in_channels = config.get("in_channels", 3)

    model = RESNET_VARIANTS[backbone_name](weights="DEFAULT" if pretrained else None)

    # Handle custom input channels
    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        model.conv1 = new_conv

    # Replace classifier head
    head_config = config.get("head", {})
    feature_dim = RESNET_FEATURE_DIMS.get(backbone_name, 512)
    num_classes = head_config.get("num_classes", 1000)

    from templates.common import build_head
    model.fc = build_head(
        "linear_cls",
        feature_dim=feature_dim,
        num_classes=num_classes,
        pooling=head_config.get("pooling", "avg"),
        dropout=head_config.get("dropout", 0.0),
    )

    return model

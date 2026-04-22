"""EfficientNet model template for image classification.

Supports efficientnet_b0.
Uses torchvision's EfficientNet implementation with configurable head.
"""

import torch.nn as nn
from torchvision import models


EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
}

EFFICIENTNET_FEATURE_DIMS = {
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
}


def build_efficientnet(config: dict) -> nn.Module:
    """Build an EfficientNet model from configuration.

    Args:
        config: Dictionary with keys:
            - backbone: One of efficientnet_b0, efficientnet_b1, efficientnet_b2.
            - pretrained: Whether to load ImageNet pretrained weights.
            - in_channels: Number of input channels (3 for RGB).
            - head: Dict with head configuration (type, num_classes, pooling, dropout).

    Returns:
        nn.Module model instance (backbone + head).
    """
    backbone_name = config.get("backbone", "efficientnet_b0")
    if backbone_name not in EFFICIENTNET_VARIANTS:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available: {sorted(EFFICIENTNET_VARIANTS.keys())}"
        )

    pretrained = config.get("pretrained", True)
    model = EFFICIENTNET_VARIANTS[backbone_name](weights="DEFAULT" if pretrained else None)

    # Replace classifier head
    head_config = config.get("head", {})
    feature_dim = EFFICIENTNET_FEATURE_DIMS.get(backbone_name, 1280)
    num_classes = head_config.get("num_classes", 1000)

    from templates.common import build_head
    model.classifier = nn.Sequential(
        nn.Dropout(p=head_config.get("dropout", 0.2), inplace=True),
        nn.Linear(feature_dim, num_classes),
    )

    return model

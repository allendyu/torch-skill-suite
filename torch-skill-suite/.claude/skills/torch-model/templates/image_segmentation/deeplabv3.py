"""DeepLabV3 model template for image segmentation.

Uses torchvision's DeepLabV3 with ResNet/MobileNet backbones.
"""

import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)


SEG_MODELS = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_mobilenet_v3_large": deeplabv3_mobilenet_v3_large,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
}


def build_deeplabv3(config: dict) -> nn.Module:
    """Build a DeepLabV3 segmentation model.

    Args:
        config: Dictionary with keys:
            - backbone: One of deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, etc.
            - pretrained: Whether to load pretrained weights.
            - in_channels: Number of input channels (3 for RGB).
            - head: Dict with num_classes.

    Returns:
        nn.Module model instance.
    """
    backbone_name = config.get("backbone", "deeplabv3_resnet50")
    pretrained = config.get("pretrained", True)
    head_config = config.get("head", {})
    num_classes = head_config.get("num_classes", 21)

    if backbone_name not in SEG_MODELS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {sorted(SEG_MODELS.keys())}")

    build_fn = SEG_MODELS[backbone_name]
    weights = "DEFAULT" if pretrained else None
    model = build_fn(weights=weights, num_classes=num_classes)

    # Handle custom input channels
    in_channels = config.get("in_channels", 3)
    if in_channels != 3 and hasattr(model, "backbone"):
        old_conv = model.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        model.backbone.conv1 = new_conv

    return model

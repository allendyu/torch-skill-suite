"""Common utility functions for model templates."""

import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    """Build a model from a configuration dictionary.

    Args:
        config: Dictionary with at least 'architecture' and head/model parameters.

    Returns:
        nn.Module model instance.
    """
    architecture = config.get("architecture", "")
    if architecture == "resnet":
        from templates.image_classification.resnet import build_resnet
        return build_resnet(config)
    elif architecture == "efficientnet":
        from templates.image_classification.efficientnet import build_efficientnet
        return build_efficientnet(config)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: resnet, efficientnet")

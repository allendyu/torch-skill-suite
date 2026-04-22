"""
Common head modules for torch-model templates.
"""

import torch
import torch.nn as nn


class LinearClsHead(nn.Module):
    """Linear classification head with optional pooling and dropout.

    Args:
        feature_dim: Input feature dimension from backbone.
        num_classes: Number of output classes.
        pooling: Pooling strategy (currently only 'avg' supported).
        dropout: Dropout rate before the final linear layer.
    """

    def __init__(self, feature_dim: int, num_classes: int, pooling: str = "avg", dropout: float = 0.0):
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pooling, dropout, and linear projection.

        Args:
            features: Tensor of shape (B, feature_dim) or (B, feature_dim, H, W).
        Returns:
            Logits of shape (B, num_classes).
        """
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        features = self.dropout(features)
        return self.fc(features)


class LinearRegressionHead(nn.Module):
    """Linear regression head for tabular regression tasks.

    Args:
        feature_dim: Input feature dimension from backbone.
        output_dim: Number of output dimensions.
        dropout: Dropout rate before the final linear layer.
    """

    def __init__(self, feature_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() > 2:
            features = features.mean(dim=[-2, -1])
        features = self.dropout(features)
        return self.fc(features)


HEAD_REGISTRY = {
    "linear_cls": LinearClsHead,
    "linear_regression": LinearRegressionHead,
}


def build_head(head_type: str, **kwargs) -> nn.Module:
    """Build a head module by type name.

    Args:
        head_type: Registered head type (e.g. 'linear_cls').
        **kwargs: Arguments forwarded to the head constructor.

    Returns:
        nn.Module head instance.
    """
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type '{head_type}'. Available: {sorted(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[head_type](**kwargs)

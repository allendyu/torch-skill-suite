"""MLP model template for tabular data.

Supports both classification and regression via head config.
"""

import torch.nn as nn


def _build_mlp_backbone(in_features, hidden_dims, dropout):
    """Build MLP feature extractor: Linear -> BN -> ReLU -> Dropout blocks."""
    layers = []
    prev_dim = in_features
    for h_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        prev_dim = h_dim
    return nn.Sequential(*layers), prev_dim


def _default_hidden_dims(in_features):
    """Heuristic for hidden layer dimensions based on input feature count."""
    if in_features <= 20:
        return [64, 32]
    elif in_features <= 100:
        return [128, 64, 32]
    else:
        return [256, 128, 64]


def build_mlp(config: dict) -> nn.Module:
    """Build an MLP model from configuration.

    Args:
        config: Dictionary with keys:
            - in_features: Number of input features.
            - hidden_dims: Optional list of hidden layer dimensions.
            - dropout: Dropout rate.
            - head: Dict with head configuration (type, num_classes/output_dim, dropout).

    Returns:
        nn.Module model instance.
    """
    in_features = config.get("in_features", 10)
    hidden_dims = config.get("hidden_dims") or _default_hidden_dims(in_features)
    dropout = config.get("dropout", 0.2)

    backbone, feature_dim = _build_mlp_backbone(in_features, hidden_dims, dropout)

    head_config = config.get("head", {})
    head_type = head_config.get("type", "linear_cls")

    from templates.common import build_head

    if head_type == "linear_cls":
        head = build_head("linear_cls", feature_dim=feature_dim,
                          num_classes=head_config.get("num_classes", 2),
                          dropout=head_config.get("dropout", 0.0))
    elif head_type == "linear_regression":
        head = build_head("linear_regression", feature_dim=feature_dim,
                          output_dim=head_config.get("output_dim", 1),
                          dropout=head_config.get("dropout", 0.0))
    else:
        raise ValueError(f"Unknown head type for MLP: {head_type}")

    class MLP(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    return MLP(backbone, head)

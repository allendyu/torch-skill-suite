"""Transformer model template for text classification.

Uses PyTorch's built-in TransformerEncoder with learned positional embeddings.
No external dependencies required — works fully offline.
"""

import math
import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """BERT-style text classifier using nn.TransformerEncoder.

    Architecture: Embedding -> PositionalEncoding -> TransformerEncoder -> CLS-pool -> Linear
    """

    def __init__(self, vocab_size=30522, hidden_size=768, num_layers=12,
                 num_heads=12, intermediate_size=3072, max_seq_length=512,
                 num_classes=2, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=intermediate_size,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        if isinstance(input_ids, dict):
            d = input_ids
            input_ids = d.get("input_ids")
            attention_mask = d.get("attention_mask")

        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Build causal mask from attention_mask
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = x[:, 0, :]  # CLS token
        return self.classifier(pooled)


# Backbone configs (vocab_size, hidden_size, num_layers, num_heads, intermediate_size)
BACKBONE_CONFIGS = {
    "bert-base-uncased": (30522, 768, 12, 12, 3072),
    "bert-tiny": (30522, 128, 4, 2, 512),
    "distilbert-base-uncased": (30522, 768, 6, 12, 3072),
}

HIDDEN_SIZES = {
    "bert-base-uncased": 768,
    "bert-tiny": 128,
    "distilbert-base-uncased": 768,
}


def build_transformer(config: dict) -> nn.Module:
    """Build a transformer-based text classification model.

    Args:
        config: Dictionary with keys:
            - backbone: Model name (e.g. 'bert-base-uncased', 'bert-tiny').
            - pretrained: Ignored (pure PyTorch, no pretrained weights).
            - head: Dict with head config (num_classes, dropout).

    Returns:
        nn.Module TextClassifier instance.
    """
    backbone_name = config.get("backbone", "bert-base-uncased")
    head_config = config.get("head", {})
    num_classes = head_config.get("num_classes", 2)
    dropout = head_config.get("dropout", 0.1)

    v, h, L, H, i = BACKBONE_CONFIGS.get(backbone_name, BACKBONE_CONFIGS["bert-tiny"])

    return TextClassifier(
        vocab_size=v, hidden_size=h, num_layers=L, num_heads=H,
        intermediate_size=i, max_seq_length=512,
        num_classes=num_classes, dropout=dropout,
    )

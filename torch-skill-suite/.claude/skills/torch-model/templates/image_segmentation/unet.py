"""UNet model template for image segmentation.

Simple encoder-decoder with skip connections.
"""

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Minimal UNet for semantic segmentation.

    Encoder: progressive Conv-BN-ReLU blocks with MaxPool downsampling.
    Decoder: upsampling + Conv-BN-ReLU blocks with skip connections.
    """

    def __init__(self, in_channels=3, num_classes=21, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, 2)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, num_classes, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


def build_unet(config: dict) -> nn.Module:
    """Build a UNet segmentation model.

    Args:
        config: Dictionary with keys:
            - backbone: Must be 'unet'.
            - pretrained: Ignored (UNet trained from scratch).
            - in_channels: Number of input channels (3 for RGB).
            - head: Dict with num_classes.

    Returns:
        nn.Module UNet instance.
    """
    in_channels = config.get("in_channels", 3)
    head_config = config.get("head", {})
    num_classes = head_config.get("num_classes", 21)
    return UNet(in_channels=in_channels, num_classes=num_classes)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class EdgeDetectionModelFixedSize(nn.Module):
    """Edge detection model that maintains input dimensions"""

    def __init__(self, in_channels=3, out_channels=1):
        super(EdgeDetectionModelFixedSize, self).__init__()

        # Encoder with padding to maintain size
        self.enc1 = self._conv_block(in_channels, 64, use_pool=False)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)

        # Decoder
        self.dec1 = self._upconv_block(256, 128)
        self.dec2 = self._upconv_block(128, 64)
        self.dec3 = self._upconv_block(64, 32)

        # Final output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch, out_ch, use_pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 64x64 -> 64x64 (no pooling)
        e2 = self.enc2(e1)  # 64x64 -> 32x32
        e3 = self.enc3(e2)  # 32x32 -> 16x16

        # Decoder
        d1 = self.dec1(e3)  # 16x16 -> 32x32
        d2 = self.dec2(d1)  # 32x32 -> 64x64
        d3 = self.dec3(d2)  # 64x64 -> 128x128? Wait, we need to fix this

        # Final output - use interpolation to get back to original size
        out = self.final(d3)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return self.sigmoid(out)
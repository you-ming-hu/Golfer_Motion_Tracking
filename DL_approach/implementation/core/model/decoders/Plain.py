
from .segmentation_models_pytorch import modules as md

import torch
import torch.nn.functional as F

from .base import BaseDecoder

class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        in_channels = [encoder_channels[-1]] + out_channels[:-1]
        self.blocks = torch.nn.Sequential(*[DecoderBlock(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])
    
    def forward(self,*features):
        x = features[-1]
        x = self.blocks(x)
        return x
    
class DecoderBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.conv1(x)
        x = self.conv2(x)
        return x


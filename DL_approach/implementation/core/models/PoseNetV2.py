import torch
import torch.nn.functional as F

from .base import BaseModel
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder

class Model(BaseModel):
    def __init__(self,encoder_name,encoder_weights="imagenet",decoder_channels=[1024, 512, 256]):
        in_channels = 3
        encoder_depth = 5
        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights)

        self.decoder = Decoder(input_channels=self.encoder.out_channels[-1],decoder_channels=decoder_channels)
        
        self.name = "u-{}".format(encoder_name)
        self.initialize()
        
class Decoder(torch.nn.Module):
    def __init__(self,input_channels,decoder_channels):
        super().__init__()
        self.out_channels = decoder_channels
        in_channels = [input_channels] + decoder_channels[:-1]
        self.blocks = torch.nn.Sequential(*[DecoderBlock(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, decoder_channels)])
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
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        return x


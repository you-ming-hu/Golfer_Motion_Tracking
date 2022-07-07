import torch

from .base import BaseModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.manet.decoder import DecoderBlock,PAB,MFAB

class Model(BaseModel):
    def __init__(
        self,
        encoder_name,
        encoder_weights="imagenet",
        encoder_freeze = False,
        decoder_channels=[1024, 512, 256]):
        
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights)
        if encoder_freeze:
            self.freeze_encoder()
        
        self.decoder = MAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels)
        
        self.name = "u-{}".format(encoder_name)
        self.initialize()


class MAnetDecoder(torch.nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        reduction=16,
        use_batchnorm=True,
        pab_channels=64,
    ):    

        super().__init__()
        
        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        
        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)  # no attention type here
        blocks = [
            MFAB(in_ch, skip_ch, out_ch, reduction=reduction, **kwargs)
            if skip_ch > 0
            else DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, self.out_channels)
        ]
        # for the last we dont have skip connection -> use simple decoder block
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

    
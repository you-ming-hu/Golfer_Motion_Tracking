import torch

from .base import BaseModel
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock,CenterBlock

class Model(BaseModel):
    def __init__(self,encoder_name,encoder_weights="imagenet",decoder_channels=[1024, 512, 256],decoder_attention_type=None):
        in_channels = 3
        encoder_depth = 5
        decoder_use_batchnorm = True
        super().__init__()
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights)

        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            center=True if encoder_name.startswith("vgg") else False)
        
        self.name = "u-{}".format(encoder_name)
        self.initialize()
        
class Decoder(torch.nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
        attention_type=None,
        center=False):
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

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = torch.nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, self.out_channels)
        ]
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

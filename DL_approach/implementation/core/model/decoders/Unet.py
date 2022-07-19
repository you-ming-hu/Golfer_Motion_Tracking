import torch

from .base import BaseDecoder

from .segmentation_models_pytorch.unet import DecoderBlock
        
class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)

        # encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

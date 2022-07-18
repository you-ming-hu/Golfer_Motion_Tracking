import torch

from .segmentation_models_pytorch.manet import DecoderBlock,PAB,MFAB

from .base import BaseDecoder

class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        reduction=16,
        use_batchnorm=True,
        pab_channels=64,
    ):    

        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)

        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            MFAB(in_ch, skip_ch, out_ch, reduction=reduction, **kwargs)
            if skip_ch > 0
            else DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]
        
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x

    
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
        self.blocks = torch.nn.ModuleList([DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, self.out_channels)])
        self.depth_skips = torch.nn.ModuleList([torch.nn.Conv2d(f,out_channels[-1],1) for f in [head_channels]+out_channels[:-1]])

    def forward(self, *features):
        features = features[::-1]

        x = features[0]
        skips = features[1:]

        output = torch.nn.functional.interpolate(self.depth_skips[0](x), scale_factor=2**len(self.out_channels), mode="bilinear",align_corners=False)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            if i < len(self.blocks) - 1:
                output = output + torch.nn.functional.interpolate(self.depth_skips[i+1](x), scale_factor=2**(len(self.out_channels)-i-1), mode="bilinear",align_corners=False)
            else:
                output = output + x
        return output
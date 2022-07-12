import re
import torch

from .base import BaseDecoder

from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
        
class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        qk_dim,
        use_batchnorm=True,
        attention_type=None):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.head = AttentionLayer(head_channels,qk_dim)
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.blocks = torch.nn.ModuleList([DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, self.out_channels)])
        self.depth_skips = torch.nn.ModuleList([torch.nn.Conv2d(f,out_channels[-1],1) for f in [head_channels]+out_channels[:-1]])

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        x = self.head(features[0])
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
    
    
class AttentionLayer(torch.nn.Module):
    def __init__(self, in_channels, qk_dim):
        super().__init__()
        self.qk_dim = qk_dim
        self.built = False
        
        self.Q = torch.nn.Conv2d(in_channels,qk_dim,1)
        self.K = torch.nn.Conv2d(in_channels,qk_dim,1)
        self.V = torch.nn.Conv2d(in_channels,in_channels,1)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        
    def init_build(self,x):
        b,c,h,w = x.shape
        PE = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,self.qk_dim)),requires_grad=True)
        self.PE = PE.transpose(0,1).view(self.PE_dim,h,w)[None,...]
        self.built = True

    def forward(self, x):
        if not self.built:
            self.init_built(x)
            
        b,c,h,w = x.shape

        q = self.Q(x) + self.PE #(B,C,H,W)
        k = self.K(x) + self.PE#(B,C,H,W)
        v = self.V(x) ##(B,O,H,W)
        
        q = q.flatten(2).transpose(1, 2) #(B,H*W,C)
        k = k.flatten(2) #(B,C,H*W)
        v = v.flatten(2).transpose(1, 2) #(B,H*W,O)
        
        qk = torch.matmul(q,k)/torch.sqrt(self.qk_dim) #(B,H*W,H*W)
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v) #(B,H*W,O)
        out = out.view(b,h,w,c).permute(0,3,1,2)
        out = self.mlp(out) + x
        return out
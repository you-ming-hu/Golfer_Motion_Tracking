import re
import torch

from .base import BaseDecoder

# from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
from segmentation_models_pytorch.base import modules as md

        
class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        squeeze = 4,
        us_patch = (9,16),
        use_batchnorm=True,
        attention_type=None):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.head = AttentionHead(head_channels,squeeze)
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type,us_patch=us_patch,us_squeeze=squeeze)
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
    
class AttentionHead(torch.nn.Module):
    def __init__(self, in_channels, squeeze):
        super().__init__()
        self.qk_dim = in_channels//squeeze
        self.built = False
        
        self.Q = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.K = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.V = torch.nn.Conv2d(in_channels,in_channels,1)
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        
    def init_build(self,x):
        b,c,h,w = x.shape
        PE = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,self.qk_dim,device=x.device)),requires_grad=True)
        self.PE = PE.transpose(0,1).view(self.qk_dim,h,w)[None,...]
        self.built = True

    def forward(self, x):
        if not self.built:
            self.init_build(x)
            
        b,c,h,w = x.shape

        q = self.Q(x) + self.PE #(B,C,H,W)
        k = self.K(x) + self.PE#(B,C,H,W)
        v = self.V(x) ##(B,O,H,W)
        
        q = q.flatten(2).transpose(1, 2) #(B,H*W,C)
        k = k.flatten(2) #(B,C,H*W)
        v = v.flatten(2).transpose(1, 2) #(B,H*W,O)
        
        qk = torch.matmul(q,k)/torch.sqrt(torch.tensor(self.qk_dim,dtype=torch.float32,device=x.device)) #(B,H*W,H*W)
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v) #(B,H*W,O)
        out = out.view(b,h,w,c).permute(0,3,1,2)
        out = self.mlp(out) + x
        return out
    
class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        us_patch,
        us_squeeze,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.upsample = AttentionUpsample(in_channels,us_patch,us_squeeze)
        
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x
    
class AttentionUpsample(torch.nn.Module):
    def __init__(self, in_channels, patch_size, squeeze):
        super().__init__()
        self.qk_dim = in_channels // squeeze
        self.patch_size = patch_size
        self.built = False
        
        self.Q = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.K = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.V = torch.nn.Conv2d(in_channels,in_channels,1)
        self.softmax = torch.nn.Softmax(dim=4)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        
    def init_build(self,x):
        h,w = self.patch_size
        pos_encode = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,self.qk_dim,device=x.device)),requires_grad=True)
        self.pos_encode = pos_encode.transpose(0,1).view(self.qk_dim,h,w)[None,...]
        self.built = True
        
    def patch_reshape(self,x,patch_size):
        b,c,h,w = x.shape
        ph,pw = patch_size
        
        x = x.view(b, c, h//ph, ph, w//pw, pw) #(B,C,h,ph,w,pw)
        x = x.permute(0,2,4,1,3,5) #(B,h,w,C,ph,pw)
        return x
    
    def out_reshape(self,out):
        ph,pw = self.patch_size
        b, h, w, n, c = out.shape
        out = out.view(b, h, w, ph*2, pw*2, c)
        out = out.permute(0,5,1,3,2,4)
        out = out.reshape(b,c,h*ph*2,w*pw*2) #(B,C,Th,Tw)
        return out

    def forward(self, x):
        if not self.built:
            self.init_build(x)
        
        k = self.K(x) #(B,C,Sh,Sw)
        k = self.patch_reshape(k,self.patch_size) #(B, sh, sw, C, ph, pw)
        k = k + self.pos_encode
        
        v = self.V(x) #(B,O,Sh,Sw)
        v = self.patch_reshape(v,self.patch_size) #(B, sh, sw, O, ph, pw)
        
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear",align_corners=False)
        
        q = self.Q(x)
        q = self.patch_reshape(q,(self.patch_size[0]*2,self.patch_size[1]*2)) #(B, sh, sw, C, ph*2, pw*2)
        q = q + torch.nn.functional.interpolate(self.pos_encode, scale_factor=2, mode="bilinear",align_corners=False)
        
        q = q.flatten(4).transpose(-1, -2) #(B, sh, sw, ph*2*pw*2, C)
        k = k.flatten(4) #(B, sh, sw, C, ph*pw)
        v = v.flatten(4).transpose(-1, -2) #(B, sh, sw, ph*pw, O)
        
        qk = torch.matmul(q,k)/torch.sqrt(torch.tensor(self.qk_dim,dtype=torch.float32,device=x.device)) #(B, sh, sw, ph*2*pw*2, ph*pw)
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v) #(B, sh, sw, ph*2*pw*2, O)
        out = self.out_reshape(out) #(B, O, Sh*2, Sw*2)
        out = self.mlp(out) + x
        return out
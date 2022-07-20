import torch
import torch.nn.functional as F
from .base import BaseDecoder

from .segmentation_models_pytorch.unet import DecoderBlock
        
class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        skip_patches = [(9,16),(9,16),(3,4),(3,4)],
        skip_channels = [1024,512,256,128],
        skip_heads = [16,8,4,2],
        unet_channels = [1024,512,256],
        unet_use_batchnorm=True,
        unet_attention_type='scse',
        out_se_reduction = 4):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        
        self.skip_layers = torch.nn.ModuleList([SkipAttention(in_ch,out_ch,p,h) for in_ch,out_ch,p,h in zip(encoder_channels,skip_channels,skip_patches,skip_heads)])
        
        unet_kwargs = dict(use_batchnorm=unet_use_batchnorm, attention_type=unet_attention_type)
        self.unet_blocks = torch.nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **unet_kwargs) 
            for in_ch, skip_ch, out_ch in zip([skip_channels[0]]+unet_channels[:-1], skip_channels[1:], unet_channels)])
        
        self.feature_pyramid = torch.nn.ModuleList([FeaturePyramid(s,ch) for s,ch in zip((3,2,1,0),[skip_channels[0]]+unet_channels)])
        self.output_se = OutputSE(sum([l.proj.out_channels for l in self.feature_pyramid]),out_channels,out_se_reduction)

    def forward(self, *features):
        features = features[::-1]
        output = []

        x = self.skip_layers[0](features[0])
        output.append(self.feature_pyramid[0](x))

        for i, block in enumerate(self.unet_blocks):
            skip = self.skip_layers[i+1](features[i+1])
            x = block(x, skip)
            output.append(self.feature_pyramid[i+1](x))
        
        output = torch.concat(output,dim=1)
        output = self.output_se(output)
        return output

class OutputSE(torch.nn.Module):
    def __init__(self, in_channel, out_channel, reduction):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channel,out_channel,1)
        self.se = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, in_channel//reduction, 1),
            torch.nn.Mish(inplace=True),
            torch.nn.Conv2d(in_channel//reduction, in_channel, 1),
            torch.nn.Sigmoid())
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, 1),
            torch.nn.Mish(inplace=True))
    
    def forward(self, x):
        out = self.se(x) * x
        out = self.mlp(out) + self.proj(x)
        return out
    
class FeaturePyramid(torch.nn.Module):
    def __init__(self, stage, in_channel):
        super().__init__()
        self.scale = 2**stage
        self.proj = torch.nn.Conv2d(in_channel,in_channel//self.scale,1,bias=False)
        
    def forward(self, x):
        x = self.proj(x)
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear",align_corners=False)
        return x

class SkipAttention(torch.nn.Module):
    def __init__(self, in_channel, out_channel, patch_size, heads):
        super().__init__()
        
        self.heads = heads
        self.pos_encoding = self.create_pos_encoding(patch_size[0],patch_size[1],out_channel,heads)
        
        self.Q = torch.nn.Conv2d(in_channel,out_channel,1)
        self.K = torch.nn.Conv2d(in_channel,out_channel,1)
        self.V = torch.nn.Conv2d(in_channel,out_channel,1)
        self.softmax = torch.nn.Softmax(dim=-2)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=1),
            )
        
    def create_pos_encoding(self,h,w,channel,heads):
        pos_encoding = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,channel//heads)),requires_grad=True)
        pos_encoding = pos_encoding.transpose(0,1).view(channel//heads,h,w) #(c,H,W)
        return pos_encoding

    def forward(self, x):
        pc,ph,pw = self.pos_encoding.shape
            
        b,c,h,w = x.shape

        q = self.Q(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6)  + self.pos_encoding 
        k = self.K(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6) + self.pos_encoding 
        v = self.V(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6)
        
        q = q.flatten(5).transpose(-1, -2) 
        k = k.flatten(5)
        v = v.flatten(5).transpose(-1, -2)
        
        qk = torch.matmul(q,k)/torch.sqrt(torch.tensor(pc,dtype=torch.float32,device=x.device))
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v)
        out = out.reshape(b, h//ph, w//pw, self.heads, ph, pw, pc).transpose(0,3,6,1,4,2,5).reshape(b,pc*self.heads,h,w)
        out = self.mlp(out) + x
        return out
    

import torch
import torch.nn.functional as F

from .base import BaseDecoder
from .segmentation_models_pytorch.unet import DecoderBlock

class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels,
        squeeze,
        head,
        use_batchnorm=True,
        attention_type='scse'):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        encoder_channels = encoder_channels[::-1]
        
        self.head = AttentionHead(encoder_channels[0],squeeze,head)
        
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.blocks = torch.nn.ModuleList(
            [DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) 
            for in_ch, skip_ch, out_ch in zip([encoder_channels[0]]+out_channels[:-1], encoder_channels[1:], out_channels)])
        
        self.feature_pyramid = torch.nn.ModuleList(
            [FeaturePyramid(s,in_ch,sk_ch,out_channels[-1])
             for s,in_ch,sk_ch in zip((3,2,1,0),[encoder_channels[0]]+out_channels,[0]+encoder_channels[1:])])

    def forward(self, *features):
        features = features[::-1]
        
        x = self.head(features[0])
        output = self.feature_pyramid[0](x,features[0])

        for i, block in enumerate(self.blocks):
            skip = features[i+1]
            x = block(x, skip)
            output += self.feature_pyramid[i+1](x,skip)
        
        return output
    
class FeaturePyramid(torch.nn.Module):
    def __init__(self, stage, in_channel, skip_channel, out_channel):
        super().__init__()
        self.scale = 2**stage
        self.proj = torch.nn.Conv2d(in_channel+skip_channel,out_channel,1)
        
    def forward(self, x, skip):
        x = torch.concat([x,skip],dim=1)
        x = self.proj(x)
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear",align_corners=False)
        return x

class AttentionHead(torch.nn.Module):
    def __init__(self, in_channels, squeeze, head):
        super().__init__()
        self.qk_dim = in_channels//squeeze
        self.head = head
        self.built = False
        
        self.Q = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.K = torch.nn.Conv2d(in_channels,self.qk_dim,1)
        self.V = torch.nn.Conv2d(in_channels,in_channels,1)
        self.softmax = torch.nn.Softmax(dim=2)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
        
    def init_build(self,x):
        b,c,h,w = x.shape
        PE = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,self.qk_dim//self.head,device=x.device)),requires_grad=True)
        self.PE = PE.transpose(0,1).view(self.qk_dim//self.head,h,w) #(c,H,W)
        self.built = True

    def forward(self, x):
        if not self.built:
            self.init_build(x)
            
        b,c,h,w = x.shape

        q = self.Q(x).view(b, self.head, self.qk_dim//self.head, h, w)  + self.PE #(B,M,c,H,W) + (c,H,W)
        k = self.K(x).view(b, self.head, self.qk_dim//self.head, h, w) + self.PE #(B,M,c,H,W) + (c,H,W)
        v = self.V(x).view(b, self.head, c//self.head, h, w) #(B,M,co,H,W)
        
        q = q.flatten(3).transpose(-1, -2) #(B,M,H*W,c)
        k = k.flatten(3) #(B,M,c,H*W)
        v = v.flatten(3).transpose(-1, -2) #(B,M,H*W,co)
        
        qk = torch.matmul(q,k)/torch.sqrt(torch.tensor(self.qk_dim/self.head,dtype=torch.float32,device=x.device)) #(B,M,H*W,H*W)
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v) #(B,M,H*W,co)
        out = out.transpose(-1,-2).reshape(b,c,h,w) #(B,M,H*W,co) -> (B,M,co,H*w) -> (B,M*co,H,W)
        out = self.mlp(out) + x
        return out
    
    

from .base import BaseEncoder
import torch
import timm

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog):
        super().__init__(None,None,aux_hog)
        self.encoder = ConvNeXt(subtype,aux_hog)
        self.out_channels = self.encoder.body.feature_info.channels()
        
class ConvNeXt(torch.nn.Module):
    def __init__(self,subtype,aux_hog):
        super().__init__()
        if aux_hog:
            in_channel = 4
        else:
            in_channel = 3
            
        self.cap = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel,64,1),
            ResBlock(64,3),
            ResBlock(64,5),
            ResBlock(64,7),
            torch.nn.Conv2d(in_channel,64,3)
        )
        self.body = timm.create_model('_'.join(['convnext',subtype]), pretrained=True, features_only=True)
        
    def forward(self,x):
        x = self.cap(x)
        outputs = self.body(x)
        return outputs
    
class ResBlock(torch.nn.Module):
    def __init__(self,in_channel,ksize):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channel,in_channel,ksize,padding='same')
        self.conv2 = torch.nn.Conv2d(in_channel,in_channel,ksize,padding='same')
        self.mish = torch.nn.Mish()
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.mish(out)
        out = self.conv2(x)
        out = out + x
        out = self.mish(out)
        return out
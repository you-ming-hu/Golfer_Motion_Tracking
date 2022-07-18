import timm
from .base import BaseEncoder

class Encoder(BaseEncoder):
    def __init__(self,subtype):
        encoder = timm.create_model('convnext_'+subtype,pretrained=True)
        out_channels = [s.blocks[-1].mlp.fc2.out_features for s in encoder.stages]
        super().__init__(encoder,out_channels)
        
    def forward(self,x):
        fms = []
        x = self.encoder.stem(x)
        for s in self.encoder.stages:
            x = s(x)
            fms.append(x)
        return fms
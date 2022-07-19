import timm
from .base import BaseEncoder

class Encoder(BaseEncoder):
    def __init__(self,subtype):
        encoder = timm.create_model('convnext_'+subtype, pretrained=True, features_only=True)
        out_channels = encoder.feature_info.channels()
        super().__init__(encoder,out_channels)
        
    def forward(self,x):
        fms = self.encoder(x)
        return fms
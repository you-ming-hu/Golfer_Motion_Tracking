import timm
import torch

class Encoder(torch.nn.Module):
    def __init__(self,subtype):
        super().__init__()
        self.encoder = timm.create_model('convnext_'+subtype,pretrained=True)
        
    def forward(self,x):
        fms = []
        x = self.encoder.stem(x)
        for s in self.encoder.stages:
            x = s(x)
            fms.append(x)
        return fms
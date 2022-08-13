from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        self.final = torch.nn.Conv2d(in_channels,num_classes,1)
        
    def forward(self,x):
        x = self.final(x)
        return x
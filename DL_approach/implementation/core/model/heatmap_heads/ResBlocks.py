from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,blocks=1):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        self.stem = torch.nn.Conv2d(in_channels,num_classes,1)
        
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(num_classes,num_classes,1),
                torch.nn.Mish(),
                torch.nn.Conv2d(num_classes,num_classes,1),
                torch.nn.Mish(),
                torch.nn.Conv2d(num_classes,num_classes,1),
                torch.nn.Mish(),
                torch.nn.Conv2d(num_classes,num_classes,1),
                torch.nn.Mish(),
                torch.nn.Conv2d(num_classes,num_classes,1)) 
            for _ in range(blocks)])
        
        self.final = torch.nn.Conv2d(num_classes,num_classes,1)
        
    def forward(self,x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x) + x
        x = self.final(x)
        return x
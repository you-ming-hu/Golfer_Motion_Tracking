from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,repeats=[3,2,1,1,1],stages=5):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        assert len(repeats) == stages
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,num_classes,1),
            torch.nn.Mish(),
            torch.nn.Conv2d(num_classes,num_classes,9,padding=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(num_classes,num_classes,7,padding=2),
            torch.nn.Mish(),
            torch.nn.Conv2d(num_classes,num_classes,5,padding=3),
            torch.nn.Mish(),
            torch.nn.Conv2d(num_classes,num_classes,3,padding=4),
            torch.nn.Mish(),
            torch.nn.Conv2d(num_classes,num_classes,1),
        )
        
        self.blocks = torch.nn.ModuleList([])
        for r,s in zip(range(1,stages+1),repeats):
            for _ in range(r):
                self.blocks.append(StageBlock(s,num_classes))
        
        self.final = torch.nn.Conv2d(num_classes,num_classes,1)
        
    def forward(self,x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x) + x
        x = self.final(x)
        return x 
    
class StageBlock(torch.nn.Module):
    def __init__(self,stage_id,channels):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(channels,channels,k,padding='same',dilation=2**stage_id-1),
                torch.nn.Mish()) 
            for k in [9,7,5,3]])
        self.final = torch.nn.Conv2d(channels,channels,1)
        
    def forward(self,x):
        for b in self.blocks:
            x = b(x) + x
        x = self.final(x)
        return x
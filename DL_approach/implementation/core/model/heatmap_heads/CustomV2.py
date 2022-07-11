from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,repeats=[3,2,1,1,1],stages=5,inner_leak=0.1,outer_leak=2):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        assert len(repeats) == stages-1
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
                self.blocks.append(StageBlock(s,num_classes,inner_leak,outer_leak))
        
    def forward(self,x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x) + x
        return x 
    
class StageBlock(torch.nn.Module):
    def __init__(self,stage_id,channels,inner_leak,outer_leak):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(channels,channels,k,padding='same',dilation=2**stage_id-1),
                torch.nn.LeakyReLU(inner_leak)) 
            for k in [9,7,5,3]])
        self.activation = torch.nn.LeakyReLU(outer_leak)
        
    def forward(self,x):
        for b in self.blocks:
            x = b(x) + x
        x = self.activation(x)
        return x
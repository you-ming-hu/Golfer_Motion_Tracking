from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,repeats=[6],stages=1):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        assert len(repeats) == stages
        self.stem = torch.nn.Conv2d(in_channels,num_classes,1)
        self.blocks = torch.nn.Sequential(*[StageBlock(s,num_classes,r) for s,r in zip(range(stages),repeats)])
        self.final = torch.nn.Conv2d(num_classes,num_classes,1)
        
    def forward(self,x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x) + x
        x = self.final(x)
        return x
    
class StageBlock(torch.nn.Module):
    def __init__(self,stage_id,channels,repeat):
        super().__init__()
        if stage_id == 0:
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(channels,channels,9,padding='same'),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,7,padding='same'),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,5,padding='same'),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,3,padding='same'),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,1)) 
                for _ in range(repeat)])
        else:
            self.blocks = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Conv2d(channels,channels,9,padding='same',dilation=2**stage_id-1),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,7,padding='same',dilation=2**stage_id-1),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,5,padding='same',dilation=2**stage_id-1),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,3,padding='same',dilation=2**stage_id-1),
                    torch.nn.Mish(),
                    torch.nn.Conv2d(channels,channels,1)) 
                for _ in range(repeat)])
        
    def forward(self,x):
        for b in self.blocks:
            x = b(x) + x
        return x
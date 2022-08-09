from .base import BaseHeatmapHead
import torch

class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,repeats=2):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        
        self.stem = torch.nn.Conv2d(in_channels,num_classes,1)
        self.blocks = ConvBlock(num_classes,repeats)
        self.final = torch.nn.Conv2d(num_classes,num_classes,1)
        
    def forward(self,x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final(x)
        return x
    
class ConvBlock(torch.nn.Module):
    def __init__(self,channels,repeat):
        super().__init__()
        
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                CrossConv(channels,9),
                CrossConv(channels,25),
                CrossConv(channels,49),
                CrossConv(channels,81)
                ) 
            for _ in range(repeat)])
        
    def forward(self,x):
        for b in self.blocks:
            x = b(x) + x
        return x
    
class CrossConv(torch.nn.Module):
    def __init__(self,channels,kernel):
        super().__init__()
        self.H = torch.nn.Conv2d(channels,channels,(kernel,1),padding='same')
        self.V = torch.nn.Conv2d(channels,channels,(1,kernel),padding='same')
        self.mish = torch.nn.Mish()
        self.conv = torch.nn.Conv2d(channels,channels,3,padding='same')
    def forward(self,x):
        x = self.H(x) + self.V(x)
        x = self.mish(x)
        x = self.conv(x)
        x = self.mish(x)
        return x
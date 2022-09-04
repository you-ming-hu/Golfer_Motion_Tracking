from .base import BaseHeatmapHead
import torch
    
class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,stem_channel=72,block_channels=[[72,72,72,72,72]]):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        
        for i in range(len(block_channels)):
            if i == 0:
                block_channels[i] = [stem_channel]+block_channels[i]
            else:
                block_channels[i] = [block_channels[i-1][-1]]+block_channels[i]
        
        self.stem = torch.nn.Conv2d(in_channels,stem_channel,1) if stem_channel is not None else torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([self.create_block(block) for block in block_channels])
        self.skips = torch.nn.ModuleList([self.create_skip(block) for block in block_channels])
        self.final = torch.nn.Conv2d(block_channels[-1][-1],num_classes,1)
            
    def create_conv2d(self,in_channels,out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,1),
            torch.nn.Mish(inplace=True))
        
    def create_skip(self,channel_list):
        if channel_list[0] == channel_list[-1]:
            return torch.nn.Identity()
        else:
            return torch.nn.Conv2d(channel_list[0],channel_list[-1],1)
        
    def create_block(self,channel_list):
        return torch.nn.Sequential(
            *[self.create_conv2d(channel_list[i],channel_list[i+1]) for i in range(len(channel_list)-2)]+[torch.nn.Conv2d(channel_list[-2],channel_list[-1],1)])
        
    def forward(self,x):
        x = self.stem(x)
        for b,s in zip(self.blocks,self.skips):
            x = b(x) + s(x)
        x = self.final(x)
        return x
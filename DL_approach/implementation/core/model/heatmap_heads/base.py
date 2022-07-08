import torch

class BaseHeatmapHead(torch.nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
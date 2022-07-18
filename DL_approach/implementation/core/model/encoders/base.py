import torch

class BaseEncoder(torch.nn.Module):
    def __init__(self,encoder,out_channels):
        super().__init__()
        self.encoder = encoder
        self.out_channels = out_channels
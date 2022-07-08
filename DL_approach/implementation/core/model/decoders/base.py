import torch

class BaseDecoder(torch.nn.Module):
    def __init__(self,encoder_channels,out_channels):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.out_channels = out_channels
from .base import BaseDetectionHead

import torch

class DetectionHead(BaseDetectionHead):
    def __init__(self,in_channels,num_classes,kernels=[(3,4),(3,4),(2,4),(2,4),(3,4)]):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        r = (num_classes*2/in_channels)**(1/len(kernels))
        channels = [round(in_channels*(r)**i) for i in range(len(kernels)+1)]
        
        mods = []
        for i,k in enumerate(kernels):
            mods.append(torch.nn.Conv2d(channels[i],channels[i+1],k))
            mods.append(torch.nn.Mish())
        mods.append(torch.nn.Flatten())
        mods.append(torch.nn.Linear(channels[-1], num_classes, bias=False))
        
        self.layers = torch.nn.Sequential(*mods)
    
    def forward(self,x):
        return self.layers(x)
        
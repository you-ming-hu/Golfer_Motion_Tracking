import torch

from .base import BaseDecoder

class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,
        out_channels
        ):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(out_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        
        self.head = DecoderHead(head_channels,[(3,4),(3,4),(2,4),(2,4),(3,4)],2,4)

        self.blocks = torch.nn.ModuleList([
            DecoderBlock(ic,sc,oc,2,4) for ic,sc,oc in zip(in_channels, skip_channels, out_channels)
            ])
        
    def forward(self,*fms):
        fms = fms[::-1]
        
        x = fms[0]
        x = self.head(x)
        
        for i,b in enumerate(self.blocks):
            x = b(x,fms[i+1])
        return x
        
class DecoderHead(torch.nn.Module):
    def __init__(self,input_channel,kernels,expand,reduce):
        super().__init__()
        block_count = len(kernels)
        in_channels = [int(input_channel*expand**(c/block_count)) for c in range(block_count)]
        out_channels = in_channels[1:] + [input_channel*expand]
        reduce_channels = [c//reduce for c in out_channels]
        
        self.block_count = block_count
        self.blocks = torch.nn.ModuleList([torch.nn.Conv2d(ic,oc,k,padding='same') for ic,oc,k in zip(in_channels,out_channels,kernels)])
        self.reduces = torch.nn.ModuleList([torch.nn.Conv2d(oc,rc,1) for oc,rc in zip(out_channels,reduce_channels)])
        self.mish = torch.nn.Mish()
        self.final = torch.nn.Conv2d(sum(reduce_channels),input_channel,1)
        
    def forward(self,x):
        outputs = []
        for i in range(self.block_count):
            x = self.blocks[i](x)
            x = self.mish(x)
            outputs.append(self.reduces[i](x))
        outputs = torch.concat(outputs,dim=1)
        outputs = self.final(outputs)
        return outputs
    
class DecoderBlock(torch.nn.Module):
    def __init__(self,in_channels, skip_channels, out_channels,conv_count,se):
        super().__init__()
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.attention1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels+skip_channels,(in_channels+skip_channels)//se,3,padding=1),
            torch.nn.Mish(),
            torch.nn.Conv2d((in_channels+skip_channels)//se,in_channels+skip_channels,1),
            torch.nn.Sigmoid()
            )
        self.convs = torch.nn.Sequential(
            ConvBlock(in_channels+skip_channels,out_channels),
            *[ConvBlock(out_channels,out_channels) for _ in range(conv_count-1)])
        self.attention2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels,out_channels//se,1),
            torch.nn.Mish(),
            torch.nn.Conv2d(out_channels//se,out_channels,1),
            torch.nn.Sigmoid()
            )
        
    def forward(self,x,skip):
        x = self.upsample(x)
        x = torch.concat([x,skip],dim=1)
        x = x * self.attention1(x)
        x = self.convs(x)
        x = x * self.attention2(x)
        return x
    
class ConvBlock(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.init_conv = torch.nn.Conv2d(in_channel,out_channel,1)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.rconvs = torch.nn.ModuleList([
            torch.nn.Conv2d(out_channel,out_channel,3,padding='same'),
            torch.nn.Conv2d(out_channel,out_channel,3,dilation=1,padding='same'),
            torch.nn.Conv2d(out_channel,out_channel,3,dilation=3,padding='same')
        ])
        self.final = torch.nn.Conv2d(out_channel*4,out_channel,1)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.mish = torch.nn.Mish()
    def forward(self,x):
        output = []
        x = self.init_conv(x)
        output.append(x)
        x = self.bn1(x)
        x = self.mish(x)
        for c in self.rconvs:
            output.append(c(x))
        x = torch.concat(output,dim=1)
        x = self.final(x)
        x = self.bn2(x)
        x = self.mish(x)
        return x
        
        
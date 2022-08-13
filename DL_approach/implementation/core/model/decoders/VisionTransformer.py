import torch
from .base import BaseDecoder

class Decoder(BaseDecoder):
    def __init__(self,encoder_channels,out_channels=[256,256,256],kernels=[4,4,4]):
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        
        layers = []
        for in_ch,out_ch,k in zip([encoder_channels[-1]]+out_channels[:-1],out_channels,kernels):
            kernel, padding, output_padding = self._get_deconv_cfg(k)
            layers.append(
                torch.nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(torch.nn.BatchNorm2d(out_ch))
            layers.append(torch.nn.ReLU(inplace=True))
        self.deconv_layers = torch.nn.Sequential(*layers)
        
        self.final_layer  = torch.nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=1)
        
        self._init_weights()
        
    def _get_deconv_cfg(self,deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')
        return deconv_kernel, padding, output_padding

    def forward(self, *x):
        x = x[0]
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def _init_weights(self):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, torch.nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, torch.nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, torch.nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                constant_init(m, 1)
                        
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        torch.nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        torch.nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias, bias)
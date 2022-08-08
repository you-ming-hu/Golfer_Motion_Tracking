import torch
from timm.models.layers import trunc_normal_
from einops import rearrange

from .base import BaseDecoder
from core.model.modules import SwinTransformerBlock

from core.dataset.common import uniform_input_image_size


IMAGE_SIZE = [uniform_input_image_size[1],uniform_input_image_size[0]]

class Decoder(BaseDecoder):
    def __init__(
        self,
        encoder_channels,out_channels,
        depths=[2, 2, 2, 2], num_heads=[24, 12, 6, 3], mlp_ratio=4.,
        patch_size=4, window_size=[9,16],
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        
        super().__init__(encoder_channels=encoder_channels,out_channels=out_channels)
        encoder_channels = encoder_channels[::-1]
        patches_resolution = [IMAGE_SIZE[0] // patch_size, IMAGE_SIZE[1] // patch_size]
        num_layers = len(encoder_channels)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.layers_up = torch.nn.ModuleList()
        self.concat_back_dim = torch.nn.ModuleList()
        for l, channel, depth, head in enumerate(encoder_channels,depths,num_heads):
            if l==0 :
                concat_linear = torch.nn.Identity()
                layer_up = PatchExpand(
                    input_resolution= (patches_resolution[0]//(2**(num_layers-1)), patches_resolution[1]//(2**(num_layers-1))),
                    dim=channel)
            else:
                concat_linear = torch.nn.Linear(channel*2,channel)
                layer_up = BasicLayer_up(
                    dim=channel,
                    input_resolution=(patches_resolution[0]//(2**(num_layers-1-l)), patches_resolution[1]//(2**(num_layers-1-l))),
                    depth=depth,
                    num_heads=head,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:l]):sum(depths[:l+1])],
                    upsample=True if (l < num_layers - 1) else False)
            
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        self.norm_up= torch.nn.LayerNorm(out_channels[-1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, *features):
        features = features[::-1]
        
        for inx, layer_up, concat_back_dim in enumerate(self.layers_up, self.concat_back_dim):
            if inx == 0:
                x = layer_up(features[0])
            else:
                x = torch.cat([x,features[inx]],-1)
                x = concat_back_dim(x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
        return x
    
class PatchExpand(torch.nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = torch.nn.Linear(dim, 2*dim, bias=False)
        self.norm = torch.nn.LayerNorm(dim // 2)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)
        return x
    
class BasicLayer_up(torch.nn.Module):
    def __init__(self,
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio,
            drop, attn_drop,drop_path, upsample):

        super().__init__()
        self.blocks = torch.nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0)|(input_resolution==window_size) else [window_size[0] // 2,window_size[1] // 2],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i])
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x
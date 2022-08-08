import torch

from .base import BaseEncoder
from core.model.modules import SwinTransformerBlock
from timm.models.layers import trunc_normal_

from core.dataset.common import uniform_input_image_size

IMAGE_SIZE = [uniform_input_image_size[1],uniform_input_image_size[0]]

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog,**kwdarg):
        super().__init__(None,None,aux_hog)
        
        self.encoder = SwinTransformer(IMAGE_SIZE, aux_hog, **kwdarg)
        
        self.out_channels = self.encoder.out_channels

class SwinTransformer(torch.nn.Module):
    def __init__(self,
            img_size, aux_hog, 
            embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.,
            patch_size=4, window_size=[9,16],
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            ape=False):
        super().__init__()

        num_layers = len(depths)

        if aux_hog:
            in_channel = 4
        else:
            in_channel = 3
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        
        self.ape = ape
        if ape:
            self.absolute_pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.out_channels = [int(embed_dim*2**k) for k in range(num_layers)]

        self.layers = torch.nn.ModuleList()
        for l, (channel, depth, head) in enumerate(zip(self.out_channels,depths,num_heads)):
            layer = BasicLayer(
                dim=channel,
                input_resolution=(patches_resolution[0] // (2 ** l),patches_resolution[1] // (2 ** l)),
                depth=depth,
                num_heads=head,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:l]):sum(depths[:l+1])],
                downsample=True if l < num_layers - 1 else False)
            self.layers.append(layer)
        
        self.norm = torch.nn.LayerNorm(self.out_channels[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x)  # B L C
        x_downsample.append(x)
        return x_downsample
    
class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]

        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x
    
class PatchMerging(torch.nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = torch.nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(torch.nn.Module):
    def __init__(self,
            dim, input_resolution, depth, num_heads, window_size, mlp_ratio,
            drop, attn_drop, drop_path, downsample):
        super().__init__()
        
        # build blocks
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

        if downsample:
            self.downsample = PatchMerging(input_resolution, dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


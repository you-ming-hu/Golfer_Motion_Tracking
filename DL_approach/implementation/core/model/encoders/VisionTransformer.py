import torch
import numpy as np

from .base import BaseEncoder
from timm.models.layers import trunc_normal_, drop_path

from core.dataset.common import uniform_input_image_size

IMAGE_SIZE = [uniform_input_image_size[1],uniform_input_image_size[0]]

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog,**kwdarg):
        super().__init__(None,None,aux_hog)
        
        self.encoder = VisionTransformer(IMAGE_SIZE, aux_hog, **kwdarg)
        
        self.stages = self.encoder.stages
        self.out_channels = self.encoder.out_channels
        
class VisionTransformer(torch.nn.Module):
    def __init__(self,
            img_size, aux_hog,
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.3):

        super().__init__()

        if aux_hog:
            in_channel = 4
        else:
            in_channel = 3
            
        self.stages = int(np.log2(patch_size))
        
        self.out_channels = [embed_dim]
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim)
        self.patch_resolution = self.patch_embed.patch_resolution
        
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = torch.nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])

        self.last_norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)

        trunc_normal_(self.pos_embed, std=.02)
        
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
           torch. nn.init.constant_(m.bias, 0)
           torch.nn.init.constant_(m.weight, 1.0)

        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, self.patch_resolution[0], self.patch_resolution[1]).contiguous()
        return [x]


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, in_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(torch.nn.Module):
    def __init__(self,dim, num_heads, qkv_bias, attn_drop, proj_drop):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(torch.nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path):
        super().__init__()
        
        self.norm1 = torch.nn.LayerNorm(dim,eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = torch.nn.LayerNorm(dim,eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size, in_chans, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
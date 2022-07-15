from sympy import perfect_power
from .base import BaseHeatmapHead
import torch
import torch.nn.functional as F
import core.dataset.common as common


class HeatmapHead(BaseHeatmapHead):
    def __init__(self,in_channels,num_classes,layers=5):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        
        total_patch_size = 16
        image_dim = 3
        head = 12
        expand = 4
        heatmap_patch = common.heatmap_downsample
        patch_size = total_patch_size//heatmap_patch
        
        pe_h = common.uniform_input_image_size[1]//total_patch_size
        pe_w = common.uniform_input_image_size[0]//total_patch_size
        pe_dim = image_dim*total_patch_size**2
        qk_dim = pe_dim
        
        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, image_dim*heatmap_patch**2, 1),
            torch.nn.Conv2d(image_dim*heatmap_patch**2, pe_dim, patch_size, stride=patch_size)
            )
        
        self.pos_encoding = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(pe_dim,pe_h,pe_w)),requires_grad=True)
        
        self.layers = torch.nn.Sequential(*[AttentionLayer(pe_dim,qk_dim,head,expand) for _ in range(layers)])
        
        self.final = torch.nn.Conv2d(pe_dim,num_classes,1)
        
    def forward(self,x):
        B,C,H,W = x.shape
        x = self.extractor(x)
        b,c,h,w = x.shape
        x = x + self.pos_encoding #(B,C,h,w)
        x = x.permute(0,2,3,1) #(B,C,h,w) -> #(B,h,w,C)
        x = x.reshape(b,h*w,c)
        
        for l in self.layers:
            x = l(x)
        #(B,h*w,c)
        x = x.reshape(b,h,w,c)
        x = x.permute(0,3,1,2)
        x = F.interpolate(x, (H,W), mode="bilinear",align_corners=False)
        x = self.final(x)
        return x
    
class AttentionLayer(torch.nn.Module):
    def __init__(self,in_dim,qk_dim,head,expand):
        super().__init__()
        self.qk_dim = qk_dim
        self.head = head
        
        self.QKV = torch.nn.Linear(in_dim,qk_dim*2+in_dim)
        self.softmax = torch.nn.Softmax(-1)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim,in_dim*expand),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim*expand,in_dim)
        )
        
    def forward(self,x):
        b,n,c = x.shape
        
        q,k,v = torch.split(self.QKV(x),[self.qk_dim,self.qk_dim,c],2)
        q = q.reshape(b, n, self.head, self.qk_dim//self.head).permute(0,2,1,3) #(B,M,N,Cqk)
        k = k.reshape(b, n, self.head, self.qk_dim//self.head).permute(0,2,1,3) #(B,M,N,Cqk)
        v = v.reshape(b, n, self.head, c//self.head).permute(0,2,1,3) #(B,M,N,Cout)
        
        score = torch.matmul(q,k.transpose(-1,-2)) / torch.sqrt(torch.tensor(self.qk_dim,dtype=x.dtype)) #(B,M,N,Cqk)*(B,M,Cqk,N) -> #(B,M,N,N)
        score = self.softmax(score)
        out = torch.matmul(score,v) #(B,M,N,N)*(B,M,N,Cout) -> (B,M,N,Cout)
        out = out.permute(0,2,1,3).reshape(b,n,c) #(B,M,N,Cout) -> (B,N,M,Cout)
        
        out = x + out
        out = self.mlp(out) + out
        return out        
from .base import BaseEncoder
import torch
import timm

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog):
        super().__init__(None,None,aux_hog)
        self.encoder = AttentionConvNeXt(subtype,aux_hog)
        self.out_channels = self.encoder.body.feature_info.channels()
        
class AttentionConvNeXt(torch.nn.Module):
    def __init__(self,subtype,aux_hog):
        if aux_hog:
            in_channel = 4
        else:
            in_channel = 3
        self.aux_hog = aux_hog
        
        self.body = timm.create_model('_'.join(['convnext',subtype]), pretrained=True, features_only=True, in_chans=in_channel)
        
        out_channels = self.body.feature_info.channels()
        self.att1 = AttentionLayer(out_channels[0])
        self.att2 = AttentionLayer(out_channels[1])
        self.att3 = AttentionLayer(out_channels[2])
        
    def forward(self,x):
        outputs = []
        x = self.body.stem_0(x)
        x = self.body.stem_1(x)
        
        x = self.body.stages_0(x)
        x = self.att1(x)
        outputs.append(x)
        
        x = self.body.stages_1(x)
        x = self.att2(x)
        outputs.append(x)
        
        x = self.body.stages_2(x)
        x = self.att3(x)
        outputs.append(x)
        
        x = self.body.stages_3(x)
        outputs.append(x)
        
        return outputs
    
class AttentionLayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, patch_size, heads):
        super().__init__()
        
        self.heads = heads
        self.pos_encoding = self.create_pos_encoding(patch_size[0],patch_size[1],out_channel,heads)
        
        self.Q = torch.nn.Conv2d(in_channel,out_channel,1)
        self.K = torch.nn.Conv2d(in_channel,out_channel,1)
        self.V = torch.nn.Conv2d(in_channel,out_channel,1)
        self.softmax = torch.nn.Softmax(dim=-2)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=1),
            )
        
    def create_pos_encoding(self,h,w,channel,heads):
        pos_encoding = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h*w,channel//heads)),requires_grad=True)
        pos_encoding = pos_encoding.transpose(0,1).view(channel//heads,h,w) #(c,H,W)
        return pos_encoding

    def forward(self, x):
        pc,ph,pw = self.pos_encoding.shape
        self.pos_encoding = self.pos_encoding.to(x.device)
        
        b,c,h,w = x.shape

        q = self.Q(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6)  + self.pos_encoding 
        k = self.K(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6) + self.pos_encoding 
        v = self.V(x).view(b, self.heads, pc, h//ph, ph, w//pw, pw).permute(0,3,5,1,2,4,6)
        
        q = q.flatten(5).transpose(-1, -2) 
        k = k.flatten(5)
        v = v.flatten(5).transpose(-1, -2)
        
        qk = torch.matmul(q,k)/torch.sqrt(torch.tensor(pc,dtype=torch.float32,device=x.device))
        qk = self.softmax(qk)
        
        out = torch.matmul(qk,v)
        out = out.reshape(b, h//ph, w//pw, self.heads, ph, pw, pc).permute(0,3,6,1,4,2,5).reshape(b,pc*self.heads,h,w)
        out = self.mlp(out) + x
        return out
    

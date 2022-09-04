from .base import BaseDetectionHead

import torch

class DetectionHead(BaseDetectionHead):
    def __init__(self,in_channels,num_classes,squeeze=2,heads=4,block_channels=[[512,256,128,64],[32,16,8],[4,2]]):
        super().__init__(in_channels=in_channels,num_classes=num_classes)
        attention_dim = in_channels//squeeze
        self.heads = heads
        self.qk_dim = attention_dim // heads
        self.pixel_count = 9*16
        
        self.PE = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(1,in_channels,self.pixel_count)),requires_grad=True)
        
        self.KV = torch.nn.Conv1d(in_channels,attention_dim*2,kernel_size=1)
        self.Q = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(1,self.heads,self.qk_dim,num_classes)),requires_grad=True)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Conv1d(attention_dim, attention_dim, kernel_size=1),
            torch.nn.Mish(),
            torch.nn.Conv1d(attention_dim, attention_dim, kernel_size=1),
            )
        
        self.final = torch.nn.Sequential(Resblocks(attention_dim,block_channels), torch.nn.Conv1d(block_channels[-1][-1], 1, kernel_size=1))
    
    def forward(self,x):
        x = x.flatten(2) #B,C,H,W -> B,C,N(H*W)
        x = x + self.PE
        k, v = torch.split(self.KV(x),self.heads*self.qk_dim,1)
        k = k.reshape(-1,self.heads,self.qk_dim,self.pixel_count) #(B,H,D,N)
        v = v.reshape(-1,self.heads,self.qk_dim,self.pixel_count) #(B,H,D,N)
        
        q = self.Q.transpose(2,3) #(B,H,D,T)->(B,H,T,D)
        qk = torch.matmul(q,k)/self.qk_dim**0.5 #(B,H,T,N)
        qk = self.softmax(qk)
        v = v.transpose(2,3)
        
        x = torch.matmul(qk,v) #(B,H,T,Dv)
        x = x.transpose(2,3) #(B,H,Dv,T)
        x = x.reshape(-1,self.heads*self.qk_dim,self.num_classes) #(B,C,T)
        
        x = self.mlp(x) + x
        
        x = self.final(x) #(B,1,T)
        x = x.squeeze(1)
        return x

class Resblocks(torch.nn.Module):
    def __init__(self,in_channels,block_channels):
        super().__init__()
        
        for i in range(len(block_channels)):
            if i == 0:
                block_channels[i] = [in_channels]+block_channels[i]
            else:
                block_channels[i] = [block_channels[i-1][-1]]+block_channels[i]
        
        self.blocks = torch.nn.ModuleList([self.create_block(block) for block in block_channels])
        self.skips = torch.nn.ModuleList([self.create_skip(block) for block in block_channels])
            
    def create_conv1d(self,in_channels,out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,out_channels,1),
            torch.nn.Mish(inplace=True))
        
    def create_skip(self,channel_list):
        if channel_list[0] == channel_list[-1]:
            return torch.nn.Identity()
        else:
            return torch.nn.Conv1d(channel_list[0],channel_list[-1],1)
        
    def create_block(self,channel_list):
        return torch.nn.Sequential(
            *[self.create_conv1d(channel_list[i],channel_list[i+1]) for i in range(len(channel_list)-2)]+[torch.nn.Conv1d(channel_list[-2],channel_list[-1],1)])
        
    def forward(self,x):
        for b,s in zip(self.blocks,self.skips):
            x = b(x) + s(x)
        return x
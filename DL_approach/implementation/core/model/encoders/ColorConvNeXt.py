from .base import BaseEncoder
import torch
import timm
import torch.distributions as D
import itertools

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog):
        super().__init__(None,None,aux_hog)
        self.encoder = ConvNeXt(subtype,aux_hog)
        self.out_channels = self.encoder.body.feature_info.channels()
        
class ConvNeXt(torch.nn.Module):
    def __init__(self,subtype,aux_hog):
        super().__init__()
        assert not aux_hog
            
        self.cap = ColorRegressor()
        self.body = timm.create_model('_'.join(['convnext',subtype]), pretrained=True, features_only=True, in_chans=125)
        
    def forward(self,x):
        x = self.cap(x)
        outputs = self.body(x)
        return outputs
    
# class ColorRegressor(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         centers = torch.tensor(list(itertools.product([0.,0.25,0.5,0.75,1.],repeat=3)),dtype=torch.float32)
#         self.centers = torch.nn.Parameter(centers)
        
#         self.relu = torch.nn.ReLU()
        
#         diag = torch.ones(len(centers),3,dtype=torch.float32)
#         self.diag = torch.nn.Parameter(diag)
        
#         sig = torch.rand(len(centers),3,3,dtype=torch.float32)*0.5
#         self.sig = torch.nn.Parameter(sig)
        
#     def forward(self,x):
#         self.centers = self.centers.to(x.device)
#         self.diag = self.diag.to(x.device)
#         self.sig = self.sig.to(x.device)
#         diag = self.relu(torch.diag_embed(self.diag))
#         cov = torch.matmul(self.sig, self.sig.transpose(-2,-1)) + diag
#         dist = D.MultivariateNormal(self.centers,cov)
#         x = x.permute(0,2,3,1)[:,:,:,None,:]
#         x = dist.log_prob(x)
#         x = torch.exp(x)
#         x = x.permute(0,3,1,2)
#         return x

# class ColorRegressor(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         centers = torch.tensor(list(itertools.product([0.,0.25,0.5,0.75,1.],repeat=3)),dtype=torch.float32)
#         self.centers = torch.nn.Parameter(centers)
        
#         self.relu = torch.nn.ReLU()
        
#         diag = torch.ones(len(centers),3,dtype=torch.float32)
#         self.diag = torch.nn.Parameter(diag)
        
        
#     def forward(self,x):
#         dist = D.Normal(self.centers,self.relu(self.diag + 1e-7))
#         x = x.permute(0,2,3,1)[:,:,:,None,:]
#         x = dist.log_prob(x)
#         x = torch.exp(x)
#         x = x.permute(0,3,1,2)
#         return x

class ColorRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = D.Normal(0,1)
        self.centers = torch.tensor(list(itertools.product([0.,0.25,0.5,0.75,1.],repeat=3)),dtype=torch.float32)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(3,8),
            torch.nn.Mish(True),
            torch.nn.Linear(8,16),
            torch.nn.Mish(True),
            torch.nn.Linear(16,8),
            torch.nn.Mish(True),
            torch.nn.Linear(8,3),
            torch.nn.Mish(True),
            )
        
    def forward(self,x):
        x = x.permute(0,2,3,1)
        x = self.proj(x)[:,:,:,None,:]
        emb = self.proj(self.centers)
        distance = torch.sqrt(((x-emb)**2).sum(dim=-1))
        x = self.gaussian.log_prob(distance)
        x = torch.exp(x)
        x = x.permute(0,3,1,2)
        return x
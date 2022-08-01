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
    
class ColorRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        centers = torch.tensor(list(itertools.product([0.,0.25,0.5,0.75,1.],repeat=3)),dtype=torch.float32)
        self.centers = torch.nn.Parameter(centers)
        
        self.relu = torch.nn.ReLU()
        cov = torch.rand(len(centers),3,3,dtype=torch.float32)*0.1 + torch.eye(3,dtype=torch.float32)
        self.cov = torch.nn.Parameter(cov)
        
    def forward(self,x):
        dist = D.MultivariateNormal(self.centers,self.relu(self.cov))
        x = x.permute(0,2,3,1)[:,:,:,None,:]
        x = dist.log_prob(x)
        x = torch.exp(x)
        x = x.permute(0,3,1,2)
        return x
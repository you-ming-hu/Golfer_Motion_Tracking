import torch
import timm

class BaseEncoder(torch.nn.Module):
    def __init__(self,encoder_name,subtype,aux_hog):
        super().__init__()
        
        if aux_hog:
            in_channel = 4
        else:
            in_channel = 3
        self.aux_hog = aux_hog
        
        if encoder_name.startswith('@'):
            raise NotImplementedError
        else:
            self.encoder = timm.create_model('_'.join([encoder_name,subtype]), pretrained=True, features_only=True, in_chans=in_channel)
            self.out_channels = self.encoder.feature_info.channels()
            
    def HOG_descriptor(self,img,ks):
        if ks == 2:
            hk = torch.tensor([1.,-1.],device=img.device)[None,None,:,None].tile(3,1,1,1)
            vk = torch.tensor([-1.,1.],device=img.device)[None,None,None,:].tile(3,1,1,1)
        else:
            hk = torch.tensor([1.,0.,-1.],device=img.device)[None,None,:,None].tile(3,1,1,1)
            vk = torch.tensor([-1.,0.,1.],device=img.device)[None,None,None,:].tile(3,1,1,1)
        
        vd_image = torch.max(torch.nn.functional.conv2d(img,vk,padding='same',groups=3),dim=1,keepdim=True)[0]
        hd_image = torch.max(torch.nn.functional.conv2d(img,hk,padding='same',groups=3),dim=1,keepdim=True)[0]
        mag_image = torch.sqrt(vd_image**2+hd_image**2)
        return mag_image
    
    def HOG_max_min_norm(self,x):
        max_v = torch.max(x,dim=[2,3],keepdim=True)[0]
        min_v = torch.min(x,dim=[2,3],keepdim=True)[0]
        x = (x-min_v)/(max_v-min_v)
        return x
    
    def forward(self,x):
        if self.aux_hog:
            hog = torch.maximum(self.HOG_descriptor(x,2),self.HOG_descriptor(x,3)) ** 0.75
            hog = self.HOG_max_min_norm(hog)
            x = torch.concat([x,hog],dim=1)
        fms = self.encoder(x)
        return fms
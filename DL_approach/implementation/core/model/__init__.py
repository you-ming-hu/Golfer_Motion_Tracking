import torch
import numpy as np
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder


import core.dataset.common as common

from . import decoders
from . import detection_heads
from . import heatmap_heads

human_keypoints_count = len(common.human_keypoints)
human_skeleton_count = len(common.human_skeleton)
golfclub_keypoints_count = len(common.golfclub_keypoints)
golfclub_skeleton_count = golfclub_keypoints_count - 1

class Model(torch.nn.Module):
    def __init__(self,structure,weights,freeze):
        super().__init__()
        self.build(**structure)
        self.initialize(weights)
        self.freeze(**freeze)
        
    def build(self,encoder,decoder,heatmap_head,detection_head):
        self.get_encoder(**encoder)
        self.get_decoder(**decoder)
        self.get_heatmap_head(**heatmap_head)
        if detection_head is not None:
            self.get_detection_head(**detection_head)
        else:
            self.detection_head = None
            
    def get_encoder(self,name,weights):
        depth = 5
        encoder = get_encoder(name=name,weights=weights,depth=depth)
        encoder.stages = depth
        self.encoder = encoder
        
    def get_decoder(self,name,out_channels,**kwdarg):
        depth = self.encoder.stages - int(np.log2(common.heatmap_downsample))
        assert len(out_channels) == depth
        decoder = getattr(decoders,name).Decoder(encoder_channels=self.encoder.out_channels,out_channels=out_channels,**kwdarg)
        self.decoder = decoder
        
    def get_heatmap_head(self,name,use_paf,**kwdarg):
        if use_paf:
            num_classes = human_keypoints_count*2 + human_skeleton_count*2 + golfclub_keypoints_count + golfclub_skeleton_count
        else:
            num_classes = human_keypoints_count*2 + golfclub_keypoints_count
        self.use_paf = use_paf
        self.heatmap_head = getattr(heatmap_heads,name).HeatmapHead(in_channels=self.decoder.out_channels[-1],num_classes=num_classes,**kwdarg)
        
    def get_detection_head(self,name,**kwdarg):
        num_classes = human_keypoints_count*3 + golfclub_keypoints_count*3 + 5
        self.detection_head = getattr(detection_heads,name).DetectionHead(in_channels=self.encoder.out_channels[-1],num_classes=num_classes,**kwdarg)
        

    def initialize(self,pretrain_weight=None):
        if pretrain_weight is not None:
            self.load_state_dict(pretrain_weight)
        else:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.heatmap_head)
            if self.detection_head is not None:
                init.initialize_head(self.detection_head)
            
    def freeze(self,encoder=False,decoder=False,detection_head=False,heatmap_head=False):
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        if detection_head:
            for param in self.detection_head.parameters():
                param.requires_grad = False
        if heatmap_head:
            for param in self.heatmap_head.parameters():
                param.requires_grad = False
        
    def intermeidate_forward(self,x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        heatmap = self.heatmap_head(decoder_output)
        print(heatmap.shape)
        if self.detection_head is not None:
            detection = self.detection_head(features[-1])
            print(detection.shape)
        else:
            detection = None
        return heatmap, detection
    
    def foramt_ouuput(self, heatmap, detection):
        output = {}
        
        if self.use_paf:
            print('use_paf')
            multi_people_heatmap,leading_role_heatmap,golfclub_heatmap = torch.split(heatmap,[
                human_keypoints_count+human_skeleton_count,
                human_keypoints_count+human_skeleton_count,
                golfclub_keypoints_count + golfclub_skeleton_count],dim=1)
            
            multi_people_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(multi_people_heatmap,[human_keypoints_count,human_skeleton_count],dim=1))}
            leading_role_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(leading_role_heatmap,[human_keypoints_count,human_skeleton_count],dim=1))}
            golfclub_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(golfclub_heatmap,[golfclub_keypoints_count,golfclub_skeleton_count],dim=1))}

            output.update({
                'multi_people_heatmap':multi_people_heatmap,
                'leading_role_heatmap':leading_role_heatmap,
                'golfclub_heatmap':golfclub_heatmap
            })
        
        else:
            print('no paf')
            multi_people_heatmap,leading_role_heatmap,golfclub_heatmap = torch.split(heatmap,[
                human_keypoints_count,
                human_keypoints_count,
                golfclub_keypoints_count],dim=1)
            
            output.update({
                'multi_people_heatmap':{'heatmap':multi_people_heatmap},
                'leading_role_heatmap':{'heatmap':leading_role_heatmap},
                'golfclub_heatmap':{'heatmap':golfclub_heatmap}
            })
        
        if detection is not None:
            print('split detection')
            leading_role_keypoints,golfclub_keypoints,leading_role_bbox = torch.split(detection,[human_keypoints_count*3,golfclub_keypoints_count*3,5],dim=-1)
            
            leading_role_keypoints_xy, leading_role_keypoints_cf = torch.split(leading_role_keypoints,[human_keypoints_count*2,human_keypoints_count],dim=-1)
            leading_role_keypoints_xy = torch.reshape(leading_role_keypoints_xy,(-1,human_keypoints_count,2))
            
            golfclub_keypoints_xy, golfclub_keypoints_cf = torch.split(golfclub_keypoints,[golfclub_keypoints_count*2,golfclub_keypoints_count],dim=-1)
            golfclub_keypoints_xy = torch.reshape(golfclub_keypoints_xy,(-1,golfclub_keypoints_count,2))
            
            leading_role_bbox_xywh,leading_role_bbox_cf = torch.split(leading_role_bbox,[4,1],dim=-1)
            leading_role_bbox_cf = torch.squeeze(leading_role_bbox_cf,dim=-1)
            
            output.update({
                'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf},
                'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf},
                'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf}
                })
        return output
    
    def forward(self,x):
        heatmap, detection = self.intermeidate_forward(x)
        output = self.foramt_ouuput(heatmap, detection)
        return output

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        heatmap, detection = self.intermeidate_forward(x)
        heatmap = torch.sigmoid(heatmap)
        if detection is not None:
            detection = torch.sigmoid(detection)
        output = self.foramt_ouuput(heatmap, detection)
        return output
    
    @torch.no_grad()
    def inference(self,output):
        return {s:torch.sigmoid(t) if not isinstance(t,dict) else {u:torch.sigmoid(v) for u,v in t.items()} for s,t in output.items()}

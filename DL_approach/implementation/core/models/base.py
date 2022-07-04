import torch
from segmentation_models_pytorch.base import initialization as init

import core.dataset.common as common

human_keypoints_count = len(common.human_keypoints)
human_skeleton_count = len(common.human_skeleton)
golfclub_keypoints_count = len(common.golfclub_keypoints)
golfclub_skeleton_count = golfclub_keypoints_count - 1

heatmap_count = human_keypoints_count*2 + human_skeleton_count*2 + golfclub_keypoints_count + golfclub_skeleton_count
target_count = human_keypoints_count*3 + golfclub_keypoints_count*3 + 5

class BaseModel(torch.nn.Module):
    
    def initialize(self):
        self.heatmap_head = HeatmapHead(self.decoder.out_channels[-1],6)
        self.detection_head = DetectionHead(self.encoder.out_channels[-1],[(3,4),(3,4),(2,4),(2,4),(3,4)])
        
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.heatmap_head)
        init.initialize_head(self.detection_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
        
    def intermeidate_forward(self,x):
        self.check_input_shape(x)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        heatmap = self.heatmap_head(decoder_output)
        detection = self.detection_head(features[-1])
        return heatmap, detection
    
    def foramt_ouuput(self, heatmap, detection):
        multi_people_heatmap,leading_role_heatmap,golfclub_heatmap = torch.split(heatmap,[
            human_keypoints_count+human_skeleton_count,
            human_keypoints_count+human_skeleton_count,
            golfclub_keypoints_count + golfclub_skeleton_count],dim=1)
        
        multi_people_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(multi_people_heatmap,[human_keypoints_count,human_skeleton_count],dim=1))}
        leading_role_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(leading_role_heatmap,[human_keypoints_count,human_skeleton_count],dim=1))}
        golfclub_heatmap = {k:v for k,v in zip(('heatmap','paf'),torch.split(golfclub_heatmap,[golfclub_keypoints_count,golfclub_skeleton_count],dim=1))}
    
        leading_role_keypoints,golfclub_keypoints,leading_role_bbox = torch.split(detection,[human_keypoints_count*3,golfclub_keypoints_count*3,5],dim=-1)
        
        leading_role_keypoints_xy, leading_role_keypoints_cf = torch.split(leading_role_keypoints,[human_keypoints_count*2,human_keypoints_count],dim=-1)
        leading_role_keypoints_xy = torch.reshape(leading_role_keypoints_xy,(-1,human_keypoints_count,2))
        
        golfclub_keypoints_xy, golfclub_keypoints_cf = torch.split(golfclub_keypoints,[golfclub_keypoints_count*2,golfclub_keypoints_count],dim=-1)
        golfclub_keypoints_xy = torch.reshape(golfclub_keypoints_xy,(-1,golfclub_keypoints_count,2))
        
        leading_role_bbox_xywh,leading_role_bbox_cf = torch.split(leading_role_bbox,[4,1],dim=-1)
        leading_role_bbox_cf = torch.squeeze(leading_role_bbox_cf,dim=-1)
        
        return {
            'multi_people_heatmap':multi_people_heatmap,
            'leading_role_heatmap':leading_role_heatmap,
            'golfclub_heatmap':golfclub_heatmap,
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf}
            }
    
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
        detection = torch.sigmoid(detection)
        output = self.foramt_ouuput(heatmap, detection)
        return output
    
    @torch.no_grad()
    def inference(self,output):
        return {s:torch.sigmoid(t) if not isinstance(t,dict) else {u:torch.sigmoid(v) for u,v in t.items()} for s,t in output.items()}

class DetectionHead(torch.nn.Sequential):
    def __init__(self,in_channel,kernels):
        r = (target_count*2/in_channel)**(1/len(kernels))
        channels = [round(in_channel*(r)**i) for i in range(len(kernels)+1)]
        
        mods = []
        for i,k in enumerate(kernels):
            mods.append(torch.nn.Conv2d(channels[i],channels[i+1],k))
            mods.append(torch.nn.Mish())
        mods.append(torch.nn.Flatten())
        mods.append(torch.nn.Linear(channels[-1], target_count, bias=False))
        super().__init__(*mods)

class HeatmapHead(torch.nn.Module):
    def __init__(self,in_channels,stages):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,heatmap_count,1),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,3,padding=1),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,5,padding=2),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,7,padding=3),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,9,padding=4),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,1),
        )
        
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
            torch.nn.Conv2d(in_channels+heatmap_count,heatmap_count,1),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,5,padding=2),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,7,padding=3),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,9,padding=4),
            torch.nn.Mish(),
            torch.nn.Conv2d(heatmap_count,heatmap_count,1),
            )
        for _ in range(stages-1)])
        
    def forward(self,fm):
        x = self.stem(fm)
        for b in self.blocks:
            x = torch.concat([x,fm],dim=1)
            x = b(x)
        return x 

    

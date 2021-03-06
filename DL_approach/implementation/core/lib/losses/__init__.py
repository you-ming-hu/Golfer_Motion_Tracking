from .losses import BaseLoss
from .losses import HeatmapUnifiedFocalLoss, PAFUnifiedFocalLoss#, AUXUnifiedFocalLoss

from .losses import KeypointsPsuedoBBox, ConfidenceFocalLoss, BBoxGIOU

class HybridLoss:
    def __init__(
        self,
        multi_people_heatmap_param,
        leading_role_heatmap_param,
        golfclub_heatmap_param,
        multi_people_paf_param,
        leading_role_paf_param,
        golfclub_paf_param,
        leading_role_keypoints_param,
        leading_role_keypoints_cf_param,
        golfclub_keypoints_param,
        golfclub_keypoints_cf_param,
        leading_role_bbox_param,
        leading_role_bbox_cf_param
        ):
        
        self.multi_people_heatmap = HeatmapUnifiedFocalLoss('multi_people_heatmap',**multi_people_heatmap_param)
        self.leading_role_heatmap = HeatmapUnifiedFocalLoss('leading_role_heatmap',**leading_role_heatmap_param)
        self.golfclub_heatmap = HeatmapUnifiedFocalLoss('golfclub_heatmap',**golfclub_heatmap_param)
        
        self.multi_people_paf = PAFUnifiedFocalLoss('multi_people_heatmap',**multi_people_paf_param)
        self.leading_role_paf = PAFUnifiedFocalLoss('leading_role_heatmap',**leading_role_paf_param)
        self.golfclub_paf = PAFUnifiedFocalLoss('golfclub_heatmap',**golfclub_paf_param)
        
        # self.multi_people_aux = AUXUnifiedFocalLoss('multi_people_heatmap',**multi_people_heatmap_param)
        # self.leading_role_aux = AUXUnifiedFocalLoss('leading_role_heatmap',**leading_role_heatmap_param)
        # self.golfclub_aux = AUXUnifiedFocalLoss('golfclub_heatmap',**golfclub_heatmap_param)
        
        self.leading_role_keypoints = KeypointsPsuedoBBox('leading_role_keypoints',**leading_role_keypoints_param)
        self.leading_role_keypoints_cf = ConfidenceFocalLoss('leading_role_keypoints',**leading_role_keypoints_cf_param)
        self.golfclub_keypoints = KeypointsPsuedoBBox('golfclub_keypoints',**golfclub_keypoints_param)
        self.golfclub_keypoints_cf = ConfidenceFocalLoss('golfclub_keypoints',**golfclub_keypoints_cf_param)
        self.leading_role_bbox = BBoxGIOU('leading_role_bbox',**leading_role_bbox_param)
        self.leading_role_bbox_cf = ConfidenceFocalLoss('leading_role_bbox',**leading_role_bbox_cf_param)
        
        self.reset_state()
        
    def reset_state(self):
        self.losses = {}
        self.losses['hybrid_loss'] = {'acc_loss':0,'acc_count':0}
        for _,l in self.__dict__.items():
            if isinstance(l,BaseLoss):
                self.losses[str(l)] = {'acc_loss':0,'acc_count':0}
                l.reset_state()
    
    def update_state(self,name,loss):
        self.losses[name]['acc_loss'] = self.losses[name]['acc_loss'] + loss.item()
        self.losses[name]['acc_count'] = self.losses[name]['acc_count'] + 1
        
    def result(self):
        return {k:v['acc_loss']/v['acc_count'] if v['acc_count']!=0 else None for k,v in self.losses.items()}
    
    def log(self,writer,data_count):
        losses = self.result()
        loss = losses['hybrid_loss']
        if loss is not None:
            writer.add_scalar('hybrid_loss',loss,data_count)
        
        for _,l in self.__dict__.items():
            if isinstance(l,BaseLoss):
                loss = losses[str(l)]
                if loss is not None:
                    writer.add_scalar(str(l),loss,data_count)
                l.log(writer,data_count)
            
        self.reset_state()
    
    def __call__(self,p,y,progression=None):
        hybrid_loss = 0
        count = 0
        drop_loss_name = []
        for name,l in self.__dict__.items():
            if isinstance(l,BaseLoss):
                loss = l(p,y)
                if loss is not None:
                    schedule = l.schedule if isinstance(l.schedule,(int,float)) else l.schedule(progression)
                    hybrid_loss = hybrid_loss + loss * schedule
                    count += 1
                    self.update_state(str(l),loss * schedule)
                    
        for name in drop_loss_name:
            self.__delattr__(name)
        
        if count == 0:
            hybrid_loss =  None
        else:
            hybrid_loss = hybrid_loss / count
            self.update_state('hybrid_loss',hybrid_loss)
            
        return hybrid_loss
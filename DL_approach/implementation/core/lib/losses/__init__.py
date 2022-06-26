from .losses import BaseLoss, HeatmapUnifiedFocalLoss, KeypointsBCE, ConfidenceFocalLoss, BBoxGIOU

class HybridLoss:
    def __init__(
        self,
        multi_human_heatmap_param,
        leading_role_heatmap_param,
        golfclub_heatmap_param,
        leading_role_keypoints_param,
        leading_role_keypoints_cf_param,
        golfclub_keypoints_param,
        golfclub_keypoints_cf_param,
        leading_role_bbox_param,
        leading_role_bbox_cf_param
        ):
        self.multi_human_heatmap = HeatmapUnifiedFocalLoss('multi_people_heatmap',**multi_human_heatmap_param)
        self.leading_role_heatmap = HeatmapUnifiedFocalLoss('leading_role_heatmap',**leading_role_heatmap_param)
        self.golfclub_heatmap = HeatmapUnifiedFocalLoss('golfclub_heatmap',**golfclub_heatmap_param)
        self.leading_role_keypoints = KeypointsBCE('leading_role_keypoints',**leading_role_keypoints_param)
        self.leading_role_keypoints_cf = ConfidenceFocalLoss('leading_role_keypoints',**leading_role_keypoints_cf_param)
        self.golfclub_keypoints = KeypointsBCE('golfclub_keypoints',**golfclub_keypoints_param)
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
        self.losses[name]['acc_loss'] = self.losses[name]['acc_loss'] + loss.cpu().numpy()
        self.losses[name]['acc_count'] = self.losses[name]['acc_count'] + 1
        
    def result(self):
        return {k:v['acc_loss']/v['acc_count'] if v['acc_count']!=0 else None for k,v in self.losses.items()}
    
    def log(self,writer,data_count):
        losses = self.result()
        for _,l in self.__dict__.items():
            if losses[str(l)] is not None:
                writer.add_scalar(str(l),losses[str(l)],data_count)
            l.log(writer,data_count)
            
        self.reset_state()
    
    def __call__(self,p,y,progression=None):
        hybrid_loss = 0
        count = 0
        
        for _,l in self.__dict__.items():
            if isinstance(l,BaseLoss):
                schedule = l.schedule if isinstance(l.schedule,(int,float)) else l.schedule(progression)
                loss = l(p,y) * schedule
                if loss is not None:
                    hybrid_loss = hybrid_loss + loss
                    count += 1
                    self.update_state(str(l),loss)
        
        if count == 0:
            hybrid_loss =  None
        else:
            self.update_state('hybrid_loss',hybrid_loss)
            
        return hybrid_loss
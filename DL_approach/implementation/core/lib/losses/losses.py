import torch
import numpy as np
import core.dataset.common as common
import core.lib.schedules.loss_schedules as loss_schedules

epsilon = 1e-7

class BaseLoss:
    def __new__(clz,name,schedule,**kwdarg):
        if schedule == 0:
            print(f'{clz.__name__} {name} schedule is 0, removed from loss') 
            return None
        else:
            return object.__new__(clz)
    
    def __init__(self,name,schedule,subclass,**kwdarg):
        for k,v in kwdarg.items():
            setattr(self,k,v)
        self.name = name
        self.schedule = schedule if isinstance(schedule,(int,float)) else eval('loss_schedules.{schedule}')
        self.subclass = subclass
        self.buffer_size = len(subclass) if subclass is not None else 1
        self.kwdarg = kwdarg
        self.reset_state()
        
    def reset_state(self):
        self.buffer = {'acc_count':np.zeros(self.buffer_size),'acc_loss':np.zeros(self.buffer_size)}
        
    def update_state(self,acc_loss,acc_count):
        self.buffer['acc_loss'] = self.buffer['acc_loss'] + acc_loss.cpu().detach().numpy()
        self.buffer['acc_count'] = self.buffer['acc_count'] + acc_count.cpu().detach().numpy()
        
    def result(self):
        output = {}
        if self.subclass is not None:
            for l,c,t in zip(self.buffer['acc_loss'],self.buffer['acc_count'],self.subclass):
                output[f'{self.__class__.__name__}({self.name}): {t}'] = l/c if c!=0 else None
        else:
            l,c = self.buffer['acc_loss'][0], self.buffer['acc_count'][0]
            output[f'{self.__class__.__name__}({self.name})'] = l/c if c!=0 else None
        return output
    
    def log(self,writer,data_count):
        losses = self.result()
        for n,l in losses.items():
            if l is not None:
                writer.add_scalar(n,l,data_count)
    
    def __str__(self):
        return self.__class__.__name__ + '('+ ','.join([self.name]+[f'{k}={v}' for k,v in self.kwdarg.items()]) + ')'
    
    def __call__(self,p,y):
        loss = self.call(p[self.name],y[self.name])
        return loss
    
    def call(self,p,y):
        raise NotImplementedError

# class HeatmapMSE(BaseLoss):
#     def __init__(self,name,schedule):
#         subclass = {'multi_people_heatmap':common.human_keypoints,'leading_role_heatmap':common.human_keypoints,'golfclub_heatmap':common.golfclub_keypoints}[name]
#         super().__init__(name,schedule,subclass)
        
#     def call(self,p,y):
#         flag = y['flag']
#         y = y['heatmap']
        
#         if torch.sum(flag) == 0:
#             loss = None
#         else:
#             p = p[flag!=0] #(B,C,H,W)
#             y = y[flag!=0]
            
#             # p = torch.sigmoid(p)
            
#             losses = torch.nn.functional.mse_loss(p,y,reduction='none')
#             losses = torch.mean(losses,dim=[2,3])
            
#             acc_loss = torch.sum(losses,axis=0)
#             acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
#             total_loss = torch.sum(acc_loss)
#             total_count = torch.sum(acc_count)
            
#             loss = total_loss/total_count
            
#             self.update_state(acc_loss,acc_count)
            
#         return loss

class UnifiedFocalLoss(BaseLoss):
    def __init__(self,name,schedule,subclass,weight,delta,gamma,warp=0,shift=0):
        assert warp>=0
        assert 0<=shift<=1
        super().__init__(name,schedule,subclass,weight=weight,delta=delta,gamma=gamma,warp=warp,shift=shift)
        
    def call(self,p,y):
        p = torch.sigmoid(p)
        p = torch.stack([1-p,p],axis=1)
        y = torch.stack([1-y,y],axis=1)
        
        if self.warp != 0:
            p = self.warp_p(p,self.warp)
        if self.shift != 0:
            p = self.shift_p(p,self.shift)
        
        tversky_losses = self.tversky_loss(p,y,self.delta,self.gamma) #(B,C,(S))
        focal_losses = self.focal_loss(p,y,self.delta,self.gamma)  #(B,(S),C)
        losses = self.weight * tversky_losses + (1-self.weight) * focal_losses  #(B,(S),C)
        return losses
    
    def warp_p(self,p,warp):
        m = 0.5
        D = 0.5
        d = p-m
        p = d * (torch.pow(torch.abs(d/D),warp)).clone().detach() + m
        return p
        
    def shift_p(self,p,shift):
        p = torch.minimum(torch.tensor(1,dtype=p.dtype,device=p.device),p+shift)
        return p
        
    def tversky_loss(self,p,y,delta,gamma):
        p = torch.clip(p, epsilon, 1. - epsilon) #(B,2,(S),C,H,W)
        tp = torch.sum(y * p, axis=[-2,-1])
        fn = torch.sum(y * (1-p), axis=[-2,-1])
        fp = torch.sum((1-y) * p, axis=[-2,-1])
        dice = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
        loss = (1-dice) * (torch.pow(1-dice, gamma)).clone().detach() #(B,2,(S),C)
        loss = torch.sum(loss,axis=1) #(B,C,(S))
        return loss
        
    def focal_loss(self,p,y,delta,gamma):
        p = torch.clip(p, epsilon, 1. - epsilon) #(B,2,(S),C,H,W)
        ce = -y*torch.log(p)
        focal = (torch.pow(1 - p, gamma)).clone().detach() * ce
        focal = torch.stack([focal[:,0,...]*(1 - delta),focal[:,1,...]*delta],axis=1) #(B,2,(S),C,H,W)
        loss = torch.sum(focal,axis=1) #(B,(S),C,H,W)
        loss = torch.mean(loss,axis=[-2,-1]) #(B,(S),C)
        return loss
    
class HeatmapUnifiedFocalLoss(UnifiedFocalLoss):
    def __init__(self,name,schedule,weight,delta,gamma):
        subclass = {'multi_people_heatmap':common.human_keypoints,'leading_role_heatmap':common.human_keypoints,'golfclub_heatmap':common.golfclub_keypoints}[name]
        super().__init__(name,schedule,subclass,weight,delta,gamma)
        
    def call(self,p,y):
        flag = y['flag']
        y = y['heatmap']
        p = p['heatmap']
        
        if torch.sum(flag) == 0:
            loss = None
        else:
            p = p[flag!=0]
            y = y[flag!=0]
            
            losses = super().call(p,y) #(B,C)
            
            acc_loss = torch.sum(losses,axis=0)
            acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
            total_loss = torch.sum(acc_loss)
            total_count = torch.sum(acc_count)
            
            loss = total_loss/total_count
            
            self.update_state(acc_loss,acc_count)
            
        return loss

class PAFUnifiedFocalLoss(UnifiedFocalLoss):
    def __init__(self,name,schedule,weight,delta,gamma):
        subclass = {'multi_people_heatmap':common.human_skeleton,'leading_role_heatmap':common.human_skeleton,'golfclub_heatmap':[(0,1)]}[name]
        super().__init__(name,schedule,subclass,weight,delta,gamma)
        
    def call(self,p,y):
        flag = y['flag']
        y = y['paf']
        p = p['paf']
        
        if torch.sum(flag) == 0:
            loss = None
        else:
            p = p[flag!=0]
            y = y[flag!=0]
            
            losses = super().call(p,y) #(B,C)
            
            acc_loss = torch.sum(losses,axis=0)
            acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
            total_loss = torch.sum(acc_loss)
            total_count = torch.sum(acc_count)
            
            loss = total_loss/total_count
            
            self.update_state(acc_loss,acc_count)
        return loss

# class AUXUnifiedFocalLoss(UnifiedFocalLoss):
#     def __init__(self,name,schedule,weight,delta,gamma):
#         super().__init__(name,schedule,['heatmap','paf'],weight,delta,gamma)
        
#     def call(self,p,y):
#         flag = y['flag']
#         y_heatmap = y['heatmap'][:,None,...]
#         x_heatmap = p['aux_heatmap']
#         y_paf = y['paf'][:,None,...]
#         x_paf = p['aux_paf']
        
#         if torch.sum(flag) == 0:
#             loss = None
#         else:
#             y_heatmap = y_heatmap[flag!=0]
#             x_heatmap = x_heatmap[flag!=0]
#             y_paf = y_paf[flag!=0]
#             x_paf = x_paf[flag!=0]
            
#             heatmap_losses = super().call(x_heatmap,y_heatmap) #(B,S,C)
#             heatmap_losses = torch.mean(heatmap_losses,axis=[1,2]) #(B)
            
#             paf_losses = super().call(x_paf,y_paf) #(B,S,C)
#             paf_losses = torch.mean(paf_losses,axis=[1,2]) #(B)
            
#             losses = torch.stack([heatmap_losses,paf_losses],dim=1)
            
#             acc_loss = torch.sum(losses,axis=0)
#             acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
#             total_loss = torch.sum(acc_loss)
#             total_count = torch.sum(acc_count)
            
#             loss = total_loss/total_count
            
#             self.update_state(acc_loss,acc_count)
#         return loss

# class HeatmapUnifiedFocalLoss(BaseLoss):
#     def __init__(self,name,schedule,weight,delta,gamma):
#         subclass = {'multi_people_heatmap':common.human_keypoints,'leading_role_heatmap':common.human_keypoints,'golfclub_heatmap':common.golfclub_keypoints}[name]
#         super().__init__(name,schedule,subclass,weight=weight,delta=delta,gamma=gamma)
        
#     def call(self,p,y):
#         flag = y['flag']
#         y = y['heatmap']
        
#         if torch.sum(flag) == 0:
#             loss = None
#         else:
#             p = p[flag!=0]
#             y = y[flag!=0]
            
#             p = torch.sigmoid(p)
#             p = torch.stack([1-p,p],axis=1)
#             y = torch.stack([1-y,y],axis=1)
            
#             tversky_losses = self.tversky_loss(p,y,self.delta,self.gamma)
#             focal_losses = self.focal_loss(p,y,self.delta,self.gamma)
#             losses = self.weight * tversky_losses + (1-self.weight) * focal_losses
            
#             acc_loss = torch.sum(losses,axis=0)
#             acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
#             total_loss = torch.sum(acc_loss)
#             total_count = torch.sum(acc_count)
            
#             loss = total_loss/total_count
            
#             self.update_state(acc_loss,acc_count)
            
#         return loss
        
#     def tversky_loss(self,p,y,delta,gamma):
#         p = torch.clip(p, epsilon, 1. - epsilon) #(B,2,C,H,W)
#         tp = torch.sum(y * p, axis=[3,4])
#         fn = torch.sum(y * (1-p), axis=[3,4])
#         fp = torch.sum((1-y) * p, axis=[3,4])
#         dice = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)
#         loss = (1-dice) * torch.pow(1-dice, gamma) #(B,2,C)
#         loss = torch.sum(loss,axis=1) #(B,C)
#         return loss
        
#     def focal_loss(self,p,y,delta,gamma):
#         p = torch.clip(p, epsilon, 1. - epsilon) #(B,2,C,H,W)
#         ce = -y*torch.log(p)
#         focal = torch.pow(1 - p, gamma) * ce
#         focal = torch.stack([focal[:,0,...]*(1 - delta),focal[:,1,...]*delta],axis=1) #(B,2,C,H,W)
#         loss = torch.sum(focal,axis=1) #(B,C,H,W)
#         loss = torch.mean(loss,axis=[2,3]) #(B,C)
#         return loss
            
class ConfidenceFocalLoss(BaseLoss):
    def __init__(self,name,schedule,delta,gamma,label_smoothing,warp=0,shift=0):
        subclass = {'leading_role_keypoints':common.human_keypoints,'golfclub_keypoints':common.golfclub_keypoints,'leading_role_bbox':None}[name]
        assert warp>=0
        assert 0<=shift<=1
        super().__init__(name,schedule,subclass,delta=delta,gamma=gamma,label_smoothing=label_smoothing,warp=warp,shift=shift)
        
    def warp_p(self,p,warp):
        m = 0.5
        D = 0.5
        d = p-m
        p = d * (torch.pow(torch.abs(d/D),warp)).clone().detach() + m
        return p
        
    def shift_p(self,p,shift):
        p = torch.minimum(torch.tensor(1,dtype=p.dtype,device=p.device),p+shift)
        return p
    
    def call(self,p,y):
        flag = y['flag']
        y = y['cf']
        p = p['cf']
        
        if torch.sum(flag) == 0:
            loss = None
        else:
            p = p[flag!=0]
            y = y[flag!=0]
            
            p = torch.sigmoid(p)
            self.update_record(p,y)
            
            y = (1-self.label_smoothing)*y + self.label_smoothing*(1-y)
            p = torch.stack([1-p,p],axis=1)
            y = torch.stack([1-y,y],axis=1)
            
            if self.warp != 0:
                p = self.warp_p(p,self.warp)
            if self.shift != 0:
                p = self.shift_p(p,self.shift)
            
            losses = self.focal_loss(p,y,self.delta,self.gamma)
            
            acc_loss = torch.sum(losses,axis=0)
            acc_count = torch.full_like(acc_loss,torch.sum(flag),dtype=torch.float32)
            
            total_loss = torch.sum(acc_loss)
            total_count = torch.sum(acc_count)
            
            loss = total_loss/total_count
            
            self.update_state(acc_loss,acc_count)
            
        return loss
    
    def reset_state(self):
        super().reset_state()
        self.record = {'p':None,'y':None}
    
    def update_record(self,p,y):
        if self.record['p'] is None:
            self.record['p'] = p.cpu().detach().numpy()
            self.record['y'] = y.cpu().detach().numpy()
        else:
            self.record['p'] = np.concatenate([self.record['p'],p.cpu().detach().numpy()],axis=0)
            self.record['y'] = np.concatenate([self.record['y'],y.cpu().detach().numpy()],axis=0)
    
    def pr_curve(self):
        output = {}
        if self.subclass is not None:
            for i,t in enumerate(self.subclass):
                p = self.record['p'][:,i] if self.record['p'] is not None else None
                y = (self.record['y'][:,i]>0.5).astype(int) if self.record['y'] is not None else None
                output[f'{self.name}_detection_pr_curve: {t}'] = (p,y)
        else:
            p = self.record['p'] if self.record['p'] is not None else None
            y = (self.record['y']>0.5).astype(int) if self.record['y'] is not None else None
            output[f'{self.name}_detection_pr_curve'] = (p,y)
        return output
    
    def log(self,writer,data_count):
        super().log(writer,data_count)
        for n,(p,y) in self.pr_curve().items():
            if y is not None:
                writer.add_pr_curve(n, y, p, data_count)
    
    def focal_loss(self,p,y,delta,gamma):
        p = torch.clip(p, epsilon, 1. - epsilon) #(B,2,C)
        ce = -y*torch.log(p)
        focal = (torch.pow(1 - p, gamma)).clone().detach() * ce
        focal = torch.stack([focal[:,0,...]*(1 - delta),focal[:,1,...]*delta],axis=1) #(B,2,C)
        loss = torch.sum(focal,axis=1) #(B,C)
        return loss

# class KeypointsRMSE(BaseLoss):
#     def __init__(self,name,schedule,tolerance):
#         subclass = {'leading_role_keypoints':common.human_keypoints,'golfclub_keypoints':common.golfclub_keypoints}[name]
#         super().__init__(name,schedule,subclass,tolerance=tolerance)
        
#     def call(self,p,y):
#         flag = y['flag']
#         cf = y['cf']
#         y = y['xy']
#         p = p['xy']
        
#         if torch.sum(flag) == 0:
#             loss = None
#         else:
#             p = p[flag!=0]
#             y = y[flag!=0]
#             cf = cf[flag!=0]
            
#             p = torch.sigmoid(p)
#             losses = torch.sqrt(torch.sum((p-y)**2,axis=2)) #(B,C)
#             losses = torch.maximum(losses-self.tolerance,torch.tensor(0))
            
#             losses = losses * cf
            
#             acc_loss = torch.sum(losses,axis=0)
#             acc_count = torch.sum(cf,axis=0)
            
#             total_loss = torch.sum(acc_loss)
#             total_count = torch.sum(acc_count)
            
#             loss = total_loss/total_count if total_count!=0 else None
            
#             self.update_state(acc_loss,acc_count)
#         return loss

class KeypointsPsuedoBBox(BaseLoss):
    def __init__(self,name,schedule,bbox_size):
        subclass = {'leading_role_keypoints':common.human_keypoints,'golfclub_keypoints':common.golfclub_keypoints}[name]
        super().__init__(name,schedule,subclass,bbox_size=bbox_size)
        
    def call(self,p,y):
        flag = y['flag']
        cf = y['cf']
        y = y['xy']
        p = p['xy']
        
        if torch.sum(flag) == 0:
            loss = None
        else:
            p = p[flag!=0]
            y = y[flag!=0]
            cf = cf[flag!=0]
            
            p = torch.sigmoid(p)
            
            losses = self.giou_loss(p,y)
            
            losses = losses * cf
            
            acc_loss = torch.sum(losses,axis=0)
            acc_count = torch.sum(cf,axis=0)
            
            total_loss = torch.sum(acc_loss)
            total_count = torch.sum(acc_count)
            
            loss = total_loss/total_count if total_count!=0 else None
            
            self.update_state(acc_loss,acc_count)
        return loss
    
    def giou_loss(self,p,y):
        image_size = torch.tensor(common.uniform_input_image_size, dtype = torch.float32)
        bbox_size = torch.tensor([self.bbox_size,self.bbox_size], dtype = torch.float32)/image_size
        bbox_size = bbox_size.to(y.device)
        bbox_size = bbox_size[None,None,:] #(B,C,(w,h))
        
        p_area = bbox_size[...,0] * bbox_size[...,1]
        y_area = bbox_size[...,0] * bbox_size[...,1]
        
        p_coor = torch.concat([p-bbox_size/2,p+bbox_size/2],axis=2)
        y_coor = torch.concat([y-bbox_size/2,y+bbox_size/2],axis=2)
        
        left_up = torch.maximum(p_coor[...,:2], y_coor[...,:2])
        right_down = torch.minimum(p_coor[...,2:], y_coor[...,2:])

        inter_section = torch.maximum(right_down - left_up, torch.tensor(0,dtype=torch.float32))
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        
        union_area = p_area + y_area - inter_area

        iou = inter_area / union_area
        iou[union_area==0] = 0

        enclose_left_up = torch.minimum(p_coor[...,:2], y_coor[...,:2])
        enclose_right_down = torch.maximum(p_coor[...,2:], y_coor[...,2:])

        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[...,0] * enclose_section[...,1]
        
        giou = (enclose_area-union_area) / enclose_area
        giou[enclose_area==0] = 0
        
        loss = iou - giou
        loss = 1 - loss #(B,C)
        return loss

class BBoxGIOU(BaseLoss):
    def __init__(self,name,schedule):
        super().__init__(name,schedule,None)
        
    def call(self,p,y):
        flag = y['flag']
        cf = y['cf']
        y = y['xywh']
        p = p['xywh']
        
        if torch.sum(flag) == 0:
            loss = None
        else:
            p = p[flag!=0]
            y = y[flag!=0]
            cf = cf[flag!=0]
            
            p = torch.sigmoid(p)
            loss = self.giou_loss(p,y)
            
            loss = loss * cf
            
            acc_loss = torch.sum(loss,axis=0)
            acc_count = torch.sum(cf,axis=0)
            
            loss = acc_loss/acc_count if acc_count!=0 else None
            
            self.update_state(acc_loss,acc_count)
        return loss #(B)
    
    def giou_loss(self,p,y):
        #(B,(x,y,w,h))
        p_area = p[:,2] * p[:,3]
        y_area = y[:,2] * y[:,3]

        p_coor = torch.concat([p[:,:2]-p[:,2:]/2, p[:,:2]+p[:,2:]/2],axis=1) #(B,(x1,y1,x2,y2))
        y_coor = torch.concat([y[:,:2]-y[:,2:]/2, y[:,:2]+y[:,2:]/2],axis=1)
        
        left_up = torch.maximum(p_coor[:,:2], y_coor[:,:2])
        right_down = torch.minimum(p_coor[:,2:], y_coor[:,2:])

        inter_section = torch.maximum(right_down - left_up, torch.tensor(0,dtype=torch.float32))
        inter_area = inter_section[:, 0] * inter_section[:, 1]
        
        union_area = p_area + y_area - inter_area

        iou = inter_area / union_area
        iou[union_area==0] = 0

        enclose_left_up = torch.minimum(p_coor[:,:2], y_coor[:,:2])
        enclose_right_down = torch.maximum(p_coor[:,2:], y_coor[:,2:])

        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[:,0] * enclose_section[:,1]
        
        giou = (enclose_area-union_area) / enclose_area
        giou[enclose_area==0] = 0
        
        loss = iou - giou
        loss = 1 - loss #(B)
        return loss
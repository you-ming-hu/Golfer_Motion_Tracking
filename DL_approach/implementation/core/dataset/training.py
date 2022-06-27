import numpy as np
from pycocotools.coco import COCO
import pathlib
import pandas as pd
import random
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from . import common
import core.lib.schedules.data_schedules as data_schedules

human_keypoints_count = len(common.human_keypoints)
golfclub_keypoints_count = len(common.golfclub_keypoints)

class DataReader:
    def __init__(self,stage,coco_path,golfer_path):
        self.coco_dummy, self.coco_human = self.get_coco_dataset(stage,coco_path)
        self.golfer = self.get_golfer_dataset(stage,golfer_path)
        self.random_state = random.Random()
        self.random_state.setstate(random.getstate())
        
    def __call__(self,golfer_coco_ratio,dummy_ratio):
        golfer_count = len(self.golfer)
        coco_count = int(golfer_count/golfer_coco_ratio)
        coco_dummy_count = int(coco_count*dummy_ratio)
        coco_human_count = coco_count-coco_dummy_count
        
        coco_human = self.random_state.sample(self.coco_human,k=coco_human_count)
        coco_dummy = self.random_state.sample(self.coco_dummy,k=coco_dummy_count)
        
        self.dataset = self.golfer + coco_human + coco_dummy
        return self
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        return self.dataset[idx]
    
    def get_golfer_dataset(self,stage,golfer_path):
        golfer_path = pathlib.Path(golfer_path,stage)
        table = pd.read_csv(golfer_path.joinpath('annotations.csv'))
        assert ['club_grip','club_head'] == common.golfclub_keypoints
        golfers = []
        
        for _,row in table.iterrows():
            img_h = row['height']
            img_w = row['width']
            
            image_info = {'file_name':golfer_path.joinpath('images',row['filename']).with_suffix('.jpg').as_posix(),'height':img_h,'width':img_w}
            
            grip_x,grip_y,head_x,head_y = row[3:].values.astype(int)
            grip_x = np.clip(grip_x,0,img_w-1)
            grip_y = np.clip(grip_y,0,img_h-1)
            head_x = np.clip(head_x,0,img_w-1)
            head_y = np.clip(head_y,0,img_h-1)
            
            grip_cf = 1 if (10<grip_x<img_w-1-10)&(10<grip_y<img_h-1-10) else 0
            head_cf = 1 if (10<head_x<img_w-1-10)&(10<head_y<img_h-1-10) else 0
            
            keypoints = np.array([[grip_x,grip_y],[head_x,head_y]])
            confidence = np.array([grip_cf,head_cf])
            golfers.append(('golfer',image_info,keypoints,confidence))
        
        return golfers

    def get_coco_dataset(self,stage,coco_path):
        coco = COCO(pathlib.Path(coco_path,'annotations',f'person_keypoints_{stage}2017.json').as_posix())
        assert coco.cats[1]['keypoints'] == common.human_keypoints
        
        coco_dummy = []
        coco_human = []
        for image_info in coco.imgs.values():
            image_info['file_name'] = pathlib.Path(coco_path,'train2017',image_info['file_name']).as_posix()
            anns = coco.getAnnIds(image_info['id'])
            
            if len(anns) == 0:
                coco_dummy.append(('coco_dummy',image_info['file_name'],image_info['width'],image_info['height']))
            else:
                anns = coco.loadAnns(anns)
                img_w = image_info['width']
                img_h = image_info['height']
                
                clean_anns = []
                for a in anns:
                    if a['iscrowd']==1:
                        continue
                    
                    x, y, w, h = np.array(a['bbox']).astype(int)
                    x = np.clip(x,0,img_w-1)
                    y = np.clip(y,0,img_h-1)
                    w = np.clip(w,0,img_w-1-x)
                    h = np.clip(h,0,img_h-1-y)
                    
                    if a['area'] > 0 and w*h>0:
                        area = a['area']
                        bbox = np.array([x,y,w,h])
                        keypoints = np.array(a['keypoints']).reshape([human_keypoints_count,3]).astype(int)
                        keypoints_xy = keypoints[:,[0,1]]
                        keypoints_cf = (keypoints[:,2]!=0).astype(int)
                        
                        if keypoints_cf.sum() != 0:
                            clean_anns.append({
                                'area':area,
                                'bbox':bbox,
                                'keypoints_xy':keypoints_xy,
                                'keypoints_cf':keypoints_cf})
                        
                if len(clean_anns) != 0:
                    
                    leading_role_id = np.argmax(np.array([x['area'] for x in clean_anns]))
                    leading_role_ann = clean_anns[leading_role_id]

                    keypoints_cf = leading_role_ann['keypoints_cf']
                    head = keypoints_cf[[0,1,2,3,4]]
                    hands = keypoints_cf[[5,6,7,8,9,10]]
                    legs = keypoints_cf[[11,12,13,14,15,16]]
                    
                    con1 = (head==0).sum() <= 1
                    con2 = (hands==0).sum() <= 2
                    con3 = (legs==0).sum() <= 2
                    
                    if con1 and con2 and con3:
                        coco_human.append(('coco_human',image_info,clean_anns,leading_role_id))
        return coco_dummy, coco_human
    
class DataProcessorBase:
    def __init__(self):
        self.divert = {'coco_human':self.process_coco_human,'coco_dummy':self.process_coco_dummy,'golfer':self.process_golfer}
        
        self.img_w, self.img_h = common.uniform_input_image_size
        
        self.hm_downsample = common.heatmap_downsample
        self.hm_w = self.img_w // self.hm_downsample
        self.hm_h = self.img_h // self.hm_downsample
        
        self.preprocess = common.preprocess
        self.postprocess = common.postprocess
        
        self.process_coco_human_param = {'w_full':0.95,'h_full':0.8,'hm_psize_ratio':1/15}
        self.process_coco_dummy_param = {}
        self.process_golfer_param = {'clubhead_ratio':1/10}
        
    
    def create_heatmap(self, center, image_size, sig): #x,y format
        x_axis = np.linspace(0, image_size[0]-1, image_size[0]) - center[0]
        y_axis = np.linspace(0, image_size[1]-1, image_size[1]) - center[1]
        xx, yy = np.meshgrid(x_axis, y_axis)
        heatmap = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return heatmap

    def rotate_with_center(self,degree,center,points):
        theta = np.radians(degree)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(R,np.array(points-center).T).T + center
    
    def __call__(self,data):
        return self.divert[data[0]](data[1:])

class DataAugProcessor(DataProcessorBase):
    def __init__(self):
        super().__init__()
        
        self.pixel_level_transform = A.Sequential([
            A.OneOrOther(A.ISONoise(p=0.5),A.GaussNoise(p=0.5),p=0.2),
            A.OneOrOther(A.Blur(5,p=0.9),A.MotionBlur(5,p=0.1),p=0.1),
            A.Downscale(0.75,0.95,p=0.1),
            A.RandomBrightnessContrast(p=0.3),
            A.OneOrOther(A.FancyPCA(alpha=0.5,p=0.4),A.HueSaturationValue(p=0.6),p=0.3),
            ],p=0.8)
        
        self.aug_coco_human_param = {
            'sacle':np.log(1.2),
            'h_full_range':(0.6,0.9),}
        self.aug_coco_dummy_param = {
            'random_resize':{'prob':0.3,'scale':np.log(1.2)},
            'random_rot':{'prob':0.3,'degree':10},
            'random_shift':{'prob':0.3,'lim':0.3}}
        self.aug_golfer_param = {
            'random_resize':{'prob':0.3,'scale':np.log(1.2)},
            'random_rot':{'prob':0.3,'degree':10},
            'random_shift':{'prob':0.6,'head_outlying_rate':0.3,'lim':0.3}}
    
    def process_coco_human(self,data):
        #load original data
        img_info,clean_anns,leading_role_id = data
        image = cv2.cvtColor(cv2.imread(img_info['file_name']),cv2.COLOR_BGR2RGB)
        bboxes_size = np.array([b['bbox'][2:] for b in clean_anns])
        keypoints_xy = np.array([k['keypoints_xy'] for k in clean_anns])
        keypoints_cf = np.array([k['keypoints_cf'] for k in clean_anns])
        #Pixel-level transforms
        image = self.pixel_level_transform(image=image)['image']
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_coco_human_param
        w_full = param['w_full']
        h_full = param['h_full']
        hm_psize_ratio = param['hm_psize_ratio']
        
        aug_param = self.aug_coco_human_param
        scale = aug_param['sacle']
        h_full_range = aug_param['h_full_range']
        
        #get affine transform
        x,y,w,h = clean_anns[leading_role_id]['bbox']
        new_w,new_h = (np.exp(np.random.uniform(-scale,scale,2)) * np.array([w,h])).astype(int)
        new_aspect = new_w/new_h
        if new_aspect > t_aspect:
            if new_w > t_w:
                new_w = int(t_w*w_full)
                new_h = int(new_w/new_aspect)
        else:
            if new_h > t_h * h_full:
                new_h = int(t_h*np.random.uniform(*h_full_range))
                new_w = int(new_h*new_aspect)
        new_x = np.random.randint(0,t_w-new_w)
        new_y = np.random.randint(0,t_h-new_h)
        src = np.array([[x,y],[x+w,y],[x,y+h]])
        dst = np.array([[new_x,new_y],[new_x+new_w,new_y],[new_x,new_y+new_h]])
        trans = cv2.getAffineTransform(np.float32(src),np.float32(dst))
        
        #transform image
        image = cv2.warpAffine(image,trans,(t_w,t_h),flags=cv2.INTER_LINEAR)
        #transform keypoints xy and cf
        new_keypoints_xy = np.concatenate([keypoints_xy.reshape(-1,2),np.ones((np.prod(keypoints_xy.shape[:2]),1))],axis=-1).T
        new_keypoints_xy = np.dot(trans, new_keypoints_xy).T
        new_keypoints_xy = new_keypoints_xy.reshape(keypoints_xy.shape).astype(int)

        inside_w = (0 < new_keypoints_xy[:,:,0]) & (new_keypoints_xy[:,:,0] < t_w)
        inside_x = (0 < new_keypoints_xy[:,:,1]) & (new_keypoints_xy[:,:,1] < t_h)
        new_keypoints_cf = ((inside_w & inside_x) & keypoints_cf.astype(bool)).astype(int)
        
        new_keypoints_xy[np.tile((new_keypoints_cf==0)[:,:,None],[1,1,2])] = 0
        #trainsform bboxes size
        new_bboxes_size = bboxes_size * np.array([new_w/w,new_h/h])
        
        hm_psize = np.fmax(1,new_bboxes_size.min(axis=-1) * hm_psize_ratio)

        #create multi_people_heatmap, leading_role_heatmap, golfclub_heatmap
        multi_people_heatmap = np.zeros((t_h,t_w,human_keypoints_count))
        leading_role_heatmap = np.zeros((t_h,t_w,human_keypoints_count))
        ann_count = len(clean_anns)
        for i in range(ann_count):
            ps = hm_psize[i]
            for j,(cf, kpt) in enumerate(zip(new_keypoints_cf[i],new_keypoints_xy[i])):
                if cf == 1:
                    new_hm = self.create_heatmap(kpt,(t_w,t_h),ps if j > 4 else ps//3+1)
                    multi_people_heatmap[:,:,j] = np.fmax(multi_people_heatmap[:,:,j], new_hm)
                    if i == leading_role_id:
                        leading_role_heatmap[:,:,j] = new_hm
        multi_people_heatmap = cv2.resize(multi_people_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        leading_role_heatmap = cv2.resize(leading_role_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        golfclub_heatmap = np.zeros((golfclub_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #create leading_role_keypoints_xy, leading_role_keypoints_cf
        leading_role_keypoints_xy = (new_keypoints_xy[leading_role_id] / np.array([t_w, t_h])).astype(np.float32)
        leading_role_keypoints_cf = (new_keypoints_cf[leading_role_id]).astype(np.float32)
        #create golfclub_keypoints_xy, golfclub_keypoints_cf
        golfclub_keypoints_xy = np.zeros((golfclub_keypoints_count,2),dtype=np.float32)
        golfclub_keypoints_cf = np.zeros(golfclub_keypoints_count,dtype=np.float32)
        #create leading_role_bbox_xywh, leading_role_bbox_cf
        leading_role_bbox_xywh = (np.array([new_x, new_y, new_w, new_h]) / np.array([t_w, t_h, t_w, t_h])).astype(np.float32)
        leading_role_bbox_cf = np.float32(1)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':1},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':1},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':1},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':1}
            }
                
    def process_coco_dummy(self,data):
        file_name,img_w,img_h = data
        image = cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2RGB)
        img_aspect = img_w/img_h
        img_center = np.array([img_w/2,img_h/2])
        img_top = np.array([img_w/2,0])
        img_left = np.array([0,img_h/2])
        #Pixel-level transforms
        image = self.pixel_level_transform(image=image)['image']
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_coco_dummy_param
        
        aug_param = self.aug_coco_dummy_param
        random_resize_prob = aug_param['random_resize']['prob']
        random_resize_scale = aug_param['random_resize']['scale']
        random_rot_prob = aug_param['random_rot']['prob']
        random_rot_degree = aug_param['random_rot']['degree']
        random_shift_prob = aug_param['random_shift']['prob']
        random_x_shift_lim = aug_param['random_shift']['lim'] * t_w
        random_y_shift_lim = aug_param['random_shift']['lim'] * t_h
        
        #fit resize
        if img_aspect > t_aspect:
            new_w = t_w
            new_h = int(np.round(new_w/img_aspect))
        elif img_aspect < t_aspect:
            new_h = t_h
            new_w = int(np.round(new_h*img_aspect))
        else:
            new_w = t_w
            new_h = t_h
        
        #random resize
        if bool(np.random.binomial(1,p=random_resize_prob)):
            new_w = new_w * np.exp(np.random.uniform(-random_resize_scale,random_resize_scale))
            new_h = new_h * np.exp(np.random.uniform(-random_resize_scale,random_resize_scale))

        new_center = np.array([t_w/2,t_h/2])
        new_top = np.array([t_w/2,t_h/2-new_h/2])
        new_right = np.array([t_w/2-new_w/2,t_h/2])

        #random rotate with center
        if bool(np.random.binomial(1,p=random_rot_prob)):
            rot = np.random.uniform(-random_rot_degree,random_rot_degree)
            new_top, new_right = self.rotate_with_center(rot,new_center,[new_top, new_right])
        
        if bool(np.random.binomial(1,p=random_shift_prob)):
            if bool(np.random.binomial(1,p=0.5)):
                x_shift = np.random.uniform(-random_x_shift_lim,random_x_shift_lim)
            else:
                x_shift = 0
            if bool(np.random.binomial(1,p=0.5)):
                y_shift = np.random.uniform(-random_y_shift_lim,random_y_shift_lim)
            else:
                y_shift = 0
            if (x_shift==0)&(y_shift==0):
                if bool(np.random.binomial(1,p=0.5)):
                    x_shift = np.random.uniform(-random_x_shift_lim,random_x_shift_lim)
                else:
                    y_shift = np.random.uniform(-random_y_shift_lim,random_y_shift_lim)
        else:
            x_shift = 0
            y_shift = 0
            
        shift = np.array([x_shift,y_shift])
        new_center = new_center + shift
        new_top = new_top + shift
        new_right = new_right + shift
        
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        transform = cv2.getAffineTransform(src,dst)
        
        #transform image
        image = cv2.warpAffine(image,transform,(t_w,t_h),flags=cv2.INTER_LINEAR)
        
        #creating multi_people_heatmap, leading_role_heatmap and creating golfclub_heatmap
        multi_people_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        leading_role_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        golfclub_heatmap = np.zeros((golfclub_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #creating leading_role_keypoints_xy and leading_role_keypoints_cf
        leading_role_keypoints_xy = np.zeros((human_keypoints_count,2),dtype=np.float32)
        leading_role_keypoints_cf = np.zeros(human_keypoints_count,dtype=np.float32)
        #creating golfclub_keypoints_xy and golfclub_keypoints_cf
        golfclub_keypoints_xy = np.zeros((golfclub_keypoints_count,2),dtype=np.float32)
        golfclub_keypoints_cf = np.zeros(golfclub_keypoints_count,dtype=np.float32)
        #creating leading_role_bbox_xywh and leading_role_bbox_cf
        leading_role_bbox_xywh = np.zeros(4,dtype=np.float32)
        leading_role_bbox_cf = np.float32(0)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':1},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':1},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':1},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':1}
            }
        

    def process_golfer(self,data):
        #loading original data
        image_info, golfclub_keypoints, glofclub_confidence = data
        image = plt.imread(image_info['file_name'])
        img_w = image_info['width']
        img_h = image_info['height']
        img_aspect = img_w/img_h
        img_center = np.array([img_w/2,img_h/2])
        img_top = np.array([img_w/2,0])
        img_left = np.array([0,img_h/2])
        #Pixel-level transforms
        image = self.pixel_level_transform(image=image)['image']
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_golfer_param
        clubhead_ratio = param['clubhead_ratio']
        
        aug_param = self.aug_golfer_param
        random_resize_prob = aug_param['random_resize']['prob']
        random_resize_scale = aug_param['random_resize']['scale']
        random_rot_prob = aug_param['random_rot']['prob']
        random_rot_degree = aug_param['random_rot']['degree']
        random_shift_prob = aug_param['random_shift']['prob']
        head_outlying_rate = aug_param['random_shift']['head_outlying_rate']
        random_x_shift_lim = aug_param['random_shift']['lim'] * t_w
        random_y_shift_lim = aug_param['random_shift']['lim'] * t_h

        #fit resize
        if img_aspect > t_aspect:
            new_w = t_w
            new_h = int(np.round(new_w/img_aspect))
        elif img_aspect < t_aspect:
            new_h = t_h
            new_w = int(np.round(new_h*img_aspect))
        else:
            new_w = t_w
            new_h = t_h
        
        #random resize
        if bool(np.random.binomial(1,p=random_resize_prob)):
            new_w = new_w * np.exp(np.random.uniform(-random_resize_scale,random_resize_scale))
            new_h = new_h * np.exp(np.random.uniform(-random_resize_scale,random_resize_scale))

        new_center = np.array([t_w/2,t_h/2])
        new_top = np.array([t_w/2,t_h/2-new_h/2])
        new_right = np.array([t_w/2-new_w/2,t_h/2])

        #random rotate with center
        if bool(np.random.binomial(1,p=random_rot_prob)):
            rot = np.random.uniform(-random_rot_degree,random_rot_degree)
            new_top, new_right = self.rotate_with_center(rot,new_center,[new_top, new_right])

        #get transform to center with ramdom resize and rotation
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        center_transform = cv2.getAffineTransform(src,dst)
        
        #update kepoints_xy maybe outlying after resize and rotate
        new_golfclub_keypoints = np.concatenate([golfclub_keypoints,np.ones((golfclub_keypoints_count,1))],axis=-1).T
        new_golfclub_keypoints = np.dot(center_transform,new_golfclub_keypoints).T

        #make sure both golfclub grip and head in the image, if the club can't fit into image make sure at least grip is inside image 
        if new_golfclub_keypoints[0,0] < 0:
            grip_x_shift = 0 - new_golfclub_keypoints[0,0]
        elif new_golfclub_keypoints[0,0] > t_w-1:
            grip_x_shift = t_w-1 - new_golfclub_keypoints[0,0]
        else:
            grip_x_shift = 0
        if new_golfclub_keypoints[0,1] < 0:
            grip_y_shift = 0 - new_golfclub_keypoints[0,1]
        elif new_golfclub_keypoints[0,1] > t_h-1:
            grip_y_shift = t_h-1 - new_golfclub_keypoints[0,1]
        else:
            grip_y_shift = 0
            
        if new_golfclub_keypoints[1,0] < 0:
            head_x_shift = 0 - new_golfclub_keypoints[1,0]
        elif new_golfclub_keypoints[1,0] > t_w-1:
            head_x_shift = t_w-1 - new_golfclub_keypoints[1,0]
        else:
            head_x_shift = 0
        if new_golfclub_keypoints[1,1] < 0:
            head_y_shift = 0 - new_golfclub_keypoints[1,1]
        elif new_golfclub_keypoints[1,1] > t_h-1:
            head_y_shift = t_h-1 - new_golfclub_keypoints[1,1]
        else:
            head_y_shift = 0

        vector = new_golfclub_keypoints[1] - new_golfclub_keypoints[0]

        if abs(vector[0]) > t_w-1:
            x_shift = new_center[0] - new_golfclub_keypoints[0,0] + np.random.uniform(-random_x_shift_lim,random_x_shift_lim)
        else:
            x_shift = max([grip_x_shift,head_x_shift],key=lambda x: abs(x))
            
        if abs(vector[1]) > t_h-1:
            y_shift = new_center[1] - new_golfclub_keypoints[0,1] + np.random.uniform(-random_y_shift_lim,random_y_shift_lim)
        else:
            y_shift = max([grip_y_shift,head_y_shift],key=lambda x: abs(x))
        
        #grip is definitely in the image after this shift, head is not in the image if and only if the club can't fit into the image
        shift = np.array([x_shift,y_shift])
        new_center = new_center + shift
        new_top = new_top + shift
        new_right = new_right + shift
        new_golfclub_keypoints = new_golfclub_keypoints + shift
        
        #update confidence
        new_grip_cf = 1 if (0<=new_golfclub_keypoints[0,0]<=t_w-1)&(0<=new_golfclub_keypoints[0,1]<=t_h-1) else 0
        new_head_cf = 1 if (0<=new_golfclub_keypoints[1,0]<=t_w-1)&(0<=new_golfclub_keypoints[1,1]<=t_h-1) else 0
        
        #head is in image
        if new_head_cf == 1:
            head_boundary = np.where(vector>0,np.array([t_w,t_h]),np.array([0,0]))
            grip_boundary = np.array([t_w,t_h]) - head_boundary
            head_random_shift = head_boundary - new_golfclub_keypoints[1]
            grip_random_shift = grip_boundary - new_golfclub_keypoints[0]
            
            #random shift (including making head outlying) or not
            if bool(np.random.binomial(1,p=random_shift_prob)):
                #random shift with golfclub head outside image
                if bool(np.random.binomial(1,p=head_outlying_rate/random_shift_prob)&(head_random_shift!=0).all()):
                    x_a = head_random_shift[0]
                    x_b = x_a + x_a/abs(x_a)*abs(vector[0])*clubhead_ratio
                    y_a = head_random_shift[1]
                    y_b = y_a + y_a/abs(y_a)*abs(vector[1])*clubhead_ratio
                    #x outside shift
                    if bool(np.random.binomial(1,p=0.5)):
                        random_x_shift = np.random.uniform(*sorted([x_a,x_b]))
                    else:
                        random_x_shift = 0
                    #y outside shift
                    if bool(np.random.binomial(1,p=0.5)):
                        random_y_shift = np.random.uniform(*sorted([y_a,y_b]))
                    else:
                        random_y_shift = 0
                    #if x,y both not shifted redo
                    if (random_x_shift==0)&(random_y_shift==0):
                        if bool(np.random.binomial(1,p=0.5)):
                            random_x_shift = np.random.uniform(*sorted([x_a,x_b]))
                        else:
                            random_y_shift = np.random.uniform(*sorted([y_a,y_b]))
                else:
                    random_x_shift = np.clip(sorted([head_random_shift[0],grip_random_shift[0]]),-random_x_shift_lim,random_x_shift_lim)
                    random_x_shift = np.random.uniform(*random_x_shift)
                    random_y_shift = np.clip(sorted([head_random_shift[1],grip_random_shift[1]]),-random_y_shift_lim,random_y_shift_lim)
                    random_y_shift = np.random.uniform(*random_y_shift)
            else:
                random_x_shift = 0
                random_y_shift = 0
                
            random_shift = np.array([random_x_shift,random_y_shift])
        #head is not in image
        else:
            #random shift (head is already outlying make sure it will not be in image after shift) or not
            if bool(np.random.binomial(1,p=random_shift_prob)):
                grip_x = new_golfclub_keypoints[0][0]
                random_x_shift_range = np.clip([-grip_x,t_w-1-grip_x],-random_x_shift_lim,random_x_shift_lim)
                grip_y = new_golfclub_keypoints[0][1]
                random_y_shift_range = np.clip([-grip_y,t_h-1-grip_y],-random_y_shift_lim,random_y_shift_lim)
                #head x in the image inferring y is not, so x can be shifted arbitrarily
                if 0<=new_golfclub_keypoints[1,0]<=t_w-1:
                    random_x_shift = np.random.uniform(*random_x_shift_range)
                else:
                    random_x_shift = 0
                #head y in the image inferring x is not, so y can be shifted arbitrarily
                if 0<=new_golfclub_keypoints[1,1]<=t_h-1:
                    random_y_shift = np.random.uniform(*random_y_shift_range)
                else:
                    random_y_shift = 0
                #both x,y outside image but leads to no random shift so shith either x or y with equal prob
                if (random_x_shift==0)&(random_y_shift==0):
                    if bool(np.random.binomial(1,p=0.5)):
                        random_x_shift = np.random.uniform(*random_x_shift_range)
                    else:
                        random_y_shift = np.random.uniform(*random_y_shift_range)
            else:
                random_x_shift = 0
                random_y_shift = 0
            
            random_shift = np.array([random_x_shift,random_y_shift])
        
        #random shift
        new_center = new_center + random_shift
        new_top = new_top + random_shift
        new_right = new_right + random_shift
        new_golfclub_keypoints = new_golfclub_keypoints + random_shift

        #get final transform after random shift
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        final_transform = cv2.getAffineTransform(src,dst)

        #update confidence becaues head maybe outside image after random shift
        new_grip_cf = 1 if (0<=new_golfclub_keypoints[0,0]<=t_w-1)&(0<=new_golfclub_keypoints[0,1]<=t_h-1) else 0
        new_head_cf = 1 if (0<=new_golfclub_keypoints[1,0]<=t_w-1)&(0<=new_golfclub_keypoints[1,1]<=t_h-1) else 0
        new_glofclub_confidence = np.array([new_grip_cf,new_head_cf])
        new_glofclub_confidence = (new_glofclub_confidence.astype(bool) & glofclub_confidence.astype(bool))
        
        new_golfclub_keypoints[np.tile((new_glofclub_confidence==0)[:,None],(1,2))] = 0

        #transform image
        image = cv2.warpAffine(image,final_transform,(t_w,t_h),flags=cv2.INTER_LINEAR)
        
        #creating multi_people_heatmap and leading_role_heatmap
        multi_people_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        leading_role_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #creating golfclub_heatmap
        hm_psize = max([1 ,(vector**2).sum()**0.5 * clubhead_ratio])
        golfclub_heatmap = np.zeros((t_h,t_w,golfclub_keypoints_count))
        for i in range(golfclub_keypoints_count):
            if new_glofclub_confidence[i] == 1:
                golfclub_heatmap[:,:,i] = self.create_heatmap(new_golfclub_keypoints[i], (t_w,t_h), hm_psize)
        golfclub_heatmap = cv2.resize(golfclub_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        #creating leading_role_keypoints_xy and leading_role_keypoints_cf
        leading_role_keypoints_xy = np.zeros((human_keypoints_count,2),dtype=np.float32)
        leading_role_keypoints_cf = np.zeros(human_keypoints_count,dtype=np.float32)
        #creating golfclub_keypoints_xy and golfclub_keypoints_cf
        golfclub_keypoints_xy = (new_golfclub_keypoints / np.array([t_w,t_h])).astype(np.float32)
        golfclub_keypoints_cf = new_glofclub_confidence.astype(np.float32)
        #creating leading_role_bbox_xywh and leading_role_bbox_cf
        leading_role_bbox_xywh = np.zeros(4,dtype=np.float32)
        leading_role_bbox_cf = np.float32(0)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':0},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':0},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':0},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':0}
            }
        
class DataNonAugProcessor(DataProcessorBase):
    def __init__(self):
        super().__init__()
    
    def process_coco_human(self,data):
        #load original data
        img_info,clean_anns,leading_role_id = data
        image = cv2.cvtColor(cv2.imread(img_info['file_name']),cv2.COLOR_BGR2RGB)
        bboxes_size = np.array([b['bbox'][2:] for b in clean_anns])
        keypoints_xy = np.array([k['keypoints_xy'] for k in clean_anns])
        keypoints_cf = np.array([k['keypoints_cf'] for k in clean_anns])
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_coco_human_param
        w_full = param['w_full']
        h_full = param['h_full']
        hm_psize_ratio = param['hm_psize_ratio']
        
        #get affine transform
        x,y,w,h = clean_anns[leading_role_id]['bbox']
        aspect = w/h
        if aspect > t_aspect:
            new_w = int(t_w*w_full)
            new_h = int(new_w/aspect)
        else:
            new_h = int(t_h*h_full)
            new_w = int(new_h*aspect)
        new_x = 0.5*(t_w-new_w)
        new_y = 0.5*(t_h-new_h)
        src = np.array([[x,y],[x+w,y],[x,y+h]])
        dst = np.array([[new_x,new_y],[new_x+new_w,new_y],[new_x,new_y+new_h]])
        trans = cv2.getAffineTransform(np.float32(src),np.float32(dst))
        
        #transform image
        image = cv2.warpAffine(image,trans,(t_w,t_h),flags=cv2.INTER_LINEAR)
        #transform keypoints xy and cf
        new_keypoints_xy = np.concatenate([keypoints_xy.reshape(-1,2),np.ones((np.prod(keypoints_xy.shape[:2]),1))],axis=-1).T
        new_keypoints_xy = np.dot(trans, new_keypoints_xy).T
        new_keypoints_xy = new_keypoints_xy.reshape(keypoints_xy.shape).astype(int)

        inside_w = (0 < new_keypoints_xy[:,:,0]) & (new_keypoints_xy[:,:,0] < t_w)
        inside_x = (0 < new_keypoints_xy[:,:,1]) & (new_keypoints_xy[:,:,1] < t_h)
        new_keypoints_cf = ((inside_w & inside_x) & keypoints_cf.astype(bool)).astype(int)
        
        new_keypoints_xy[np.tile((new_keypoints_cf==0)[:,:,None],[1,1,2])] = 0
        #trainsform bboxes size
        new_bboxes_size = bboxes_size * np.array([new_w/w,new_h/h])
        
        hm_psize = np.fmax(1,new_bboxes_size.min(axis=-1) * hm_psize_ratio)

        #create multi_people_heatmap, leading_role_heatmap, golfclub_heatmap
        multi_people_heatmap = np.zeros((t_h,t_w,human_keypoints_count))
        leading_role_heatmap = np.zeros((t_h,t_w,human_keypoints_count))
        ann_count = len(clean_anns)
        for i in range(ann_count):
            ps = hm_psize[i]
            for j,(cf, kpt) in enumerate(zip(new_keypoints_cf[i],new_keypoints_xy[i])):
                if cf == 1:
                    new_hm = self.create_heatmap(kpt,(t_w,t_h),ps if j > 4 else ps//3+1)
                    multi_people_heatmap[:,:,j] = np.fmax(multi_people_heatmap[:,:,j], new_hm)
                    if i == leading_role_id:
                        leading_role_heatmap[:,:,j] = new_hm
        multi_people_heatmap = cv2.resize(multi_people_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        leading_role_heatmap = cv2.resize(leading_role_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        golfclub_heatmap = np.zeros((golfclub_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #create leading_role_keypoints_xy, leading_role_keypoints_cf
        leading_role_keypoints_xy = (new_keypoints_xy[leading_role_id] / np.array([t_w, t_h])).astype(np.float32)
        leading_role_keypoints_cf = (new_keypoints_cf[leading_role_id]).astype(np.float32)
        #create golfclub_keypoints_xy, golfclub_keypoints_cf
        golfclub_keypoints_xy = np.zeros((golfclub_keypoints_count,2),dtype=np.float32)
        golfclub_keypoints_cf = np.zeros(golfclub_keypoints_count,dtype=np.float32)
        #create leading_role_bbox_xywh, leading_role_bbox_cf
        leading_role_bbox_xywh = (np.array([new_x, new_y, new_w, new_h]) / np.array([t_w, t_h, t_w, t_h])).astype(np.float32)
        leading_role_bbox_cf = np.float32(1)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':1},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':1},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':1},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':1}
            }
                
    def process_coco_dummy(self,data):
        file_name,img_w,img_h = data
        image = cv2.cvtColor(cv2.imread(file_name),cv2.COLOR_BGR2RGB)
        img_aspect = img_w/img_h
        img_center = np.array([img_w/2,img_h/2])
        img_top = np.array([img_w/2,0])
        img_left = np.array([0,img_h/2])
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_coco_dummy_param
        
        #fit resize
        if img_aspect > t_aspect:
            new_w = t_w
            new_h = int(np.round(new_w/img_aspect))
        elif img_aspect < t_aspect:
            new_h = t_h
            new_w = int(np.round(new_h*img_aspect))
        else:
            new_w = t_w
            new_h = t_h

        new_center = np.array([t_w/2,t_h/2])
        new_top = np.array([t_w/2,t_h/2-new_h/2])
        new_right = np.array([t_w/2-new_w/2,t_h/2])
        
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        transform = cv2.getAffineTransform(src,dst)
        
        #transform image
        image = cv2.warpAffine(image,transform,(t_w,t_h),flags=cv2.INTER_LINEAR)
        
        #creating multi_people_heatmap, leading_role_heatmap and creating golfclub_heatmap
        multi_people_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        leading_role_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        golfclub_heatmap = np.zeros((golfclub_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #creating leading_role_keypoints_xy and leading_role_keypoints_cf
        leading_role_keypoints_xy = np.zeros((human_keypoints_count,2),dtype=np.float32)
        leading_role_keypoints_cf = np.zeros(human_keypoints_count,dtype=np.float32)
        #creating golfclub_keypoints_xy and golfclub_keypoints_cf
        golfclub_keypoints_xy = np.zeros((golfclub_keypoints_count,2),dtype=np.float32)
        golfclub_keypoints_cf = np.zeros(golfclub_keypoints_count,dtype=np.float32)
        #creating leading_role_bbox_xywh and leading_role_bbox_cf
        leading_role_bbox_xywh = np.zeros(4,dtype=np.float32)
        leading_role_bbox_cf = np.float32(0)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':1},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':1},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':1},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':1}
            }
        

    def process_golfer(self,data):
        #loading original data
        image_info, golfclub_keypoints, glofclub_confidence = data
        image = plt.imread(image_info['file_name'])
        img_w = image_info['width']
        img_h = image_info['height']
        img_aspect = img_w/img_h
        img_center = np.array([img_w/2,img_h/2])
        img_top = np.array([img_w/2,0])
        img_left = np.array([0,img_h/2])
        #preprocess: CLACHE
        image = self.preprocess(image=image)['image']
        #Spatial-level transforms
        t_w = self.img_w
        t_h = self.img_h
        t_aspect = t_w/t_h
        
        hm_h = self.hm_h
        hm_w = self.hm_w
        
        param = self.process_golfer_param
        clubhead_ratio = param['clubhead_ratio']

        #fit resize
        if img_aspect > t_aspect:
            new_w = t_w
            new_h = int(np.round(new_w/img_aspect))
        elif img_aspect < t_aspect:
            new_h = t_h
            new_w = int(np.round(new_h*img_aspect))
        else:
            new_w = t_w
            new_h = t_h

        new_center = np.array([t_w/2,t_h/2])
        new_top = np.array([t_w/2,t_h/2-new_h/2])
        new_right = np.array([t_w/2-new_w/2,t_h/2])

        #get transform to center
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        center_transform = cv2.getAffineTransform(src,dst)
        
        #update kepoints_xy maybe outlying after resize
        new_golfclub_keypoints = np.concatenate([golfclub_keypoints,np.ones((golfclub_keypoints_count,1))],axis=-1).T
        new_golfclub_keypoints = np.dot(center_transform,new_golfclub_keypoints).T

        #make sure both golfclub grip and head in the image, if the club can't fit into image make sure at least grip is inside image 
        if new_golfclub_keypoints[0,0] < 0:
            grip_x_shift = 0 - new_golfclub_keypoints[0,0]
        elif new_golfclub_keypoints[0,0] > t_w-1:
            grip_x_shift = t_w-1 - new_golfclub_keypoints[0,0]
        else:
            grip_x_shift = 0
        if new_golfclub_keypoints[0,1] < 0:
            grip_y_shift = 0 - new_golfclub_keypoints[0,1]
        elif new_golfclub_keypoints[0,1] > t_h-1:
            grip_y_shift = t_h-1 - new_golfclub_keypoints[0,1]
        else:
            grip_y_shift = 0
            
        if new_golfclub_keypoints[1,0] < 0:
            head_x_shift = 0 - new_golfclub_keypoints[1,0]
        elif new_golfclub_keypoints[1,0] > t_w-1:
            head_x_shift = t_w-1 - new_golfclub_keypoints[1,0]
        else:
            head_x_shift = 0
        if new_golfclub_keypoints[1,1] < 0:
            head_y_shift = 0 - new_golfclub_keypoints[1,1]
        elif new_golfclub_keypoints[1,1] > t_h-1:
            head_y_shift = t_h-1 - new_golfclub_keypoints[1,1]
        else:
            head_y_shift = 0

        vector = new_golfclub_keypoints[1] - new_golfclub_keypoints[0]

        if abs(vector[0]) > t_w-1:
            x_shift = new_center[0] - new_golfclub_keypoints[0,0]
        else:
            x_shift = max([grip_x_shift,head_x_shift],key=lambda x: abs(x))
            
        if abs(vector[1]) > t_h-1:
            y_shift = new_center[1] - new_golfclub_keypoints[0,1]
        else:
            y_shift = max([grip_y_shift,head_y_shift],key=lambda x: abs(x))
        
        #grip is definitely in the image after this shift, head is not in the image if and only if the club can't fit into the image
        shift = np.array([x_shift,y_shift])
        new_center = new_center + shift
        new_top = new_top + shift
        new_right = new_right + shift
        new_golfclub_keypoints = new_golfclub_keypoints + shift
        
        #update confidence becaues head maybe outside image
        new_grip_cf = 1 if (0<=new_golfclub_keypoints[0,0]<=t_w-1)&(0<=new_golfclub_keypoints[0,1]<=t_h-1) else 0
        new_head_cf = 1 if (0<=new_golfclub_keypoints[1,0]<=t_w-1)&(0<=new_golfclub_keypoints[1,1]<=t_h-1) else 0
        new_glofclub_confidence = np.array([new_grip_cf,new_head_cf])
        new_glofclub_confidence = (new_glofclub_confidence.astype(bool) & glofclub_confidence.astype(bool))
        
        new_golfclub_keypoints[np.tile((new_glofclub_confidence==0)[:,None],(1,2))] = 0
        
        #get final transform
        src = np.array([img_center,img_top,img_left],dtype=np.float32)
        dst = np.array([new_center,new_top,new_right],dtype=np.float32)
        final_transform = cv2.getAffineTransform(src,dst)

        #transform image
        image = cv2.warpAffine(image,final_transform,(t_w,t_h),flags=cv2.INTER_LINEAR)
        
        #creating multi_people_heatmap and leading_role_heatmap
        multi_people_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        leading_role_heatmap = np.zeros((human_keypoints_count,hm_h,hm_w),dtype=np.float32)
        #creating golfclub_heatmap
        hm_psize = max([1 ,(vector**2).sum()**0.5 * clubhead_ratio])
        golfclub_heatmap = np.zeros((t_h,t_w,golfclub_keypoints_count))
        for i in range(golfclub_keypoints_count):
            if new_glofclub_confidence[i] == 1:
                golfclub_heatmap[:,:,i] = self.create_heatmap(new_golfclub_keypoints[i], (t_w,t_h), hm_psize)
        golfclub_heatmap = cv2.resize(golfclub_heatmap,(hm_w,hm_h)).transpose([2,0,1]).astype(np.float32)
        #creating leading_role_keypoints_xy and leading_role_keypoints_cf
        leading_role_keypoints_xy = np.zeros((human_keypoints_count,2),dtype=np.float32)
        leading_role_keypoints_cf = np.zeros(human_keypoints_count,dtype=np.float32)
        #creating golfclub_keypoints_xy and golfclub_keypoints_cf
        golfclub_keypoints_xy = (new_golfclub_keypoints / np.array([t_w,t_h])).astype(np.float32)
        golfclub_keypoints_cf = new_glofclub_confidence.astype(np.float32)
        #creating leading_role_bbox_xywh and leading_role_bbox_cf
        leading_role_bbox_xywh = np.zeros(4,dtype=np.float32)
        leading_role_bbox_cf = np.float32(0)
        
        #postprocess : ToFloat, Normalize, ToTensor
        image = self.postprocess(image=image)['image']
        
        return {
            'image':image,
            'multi_people_heatmap':{'heatmap':multi_people_heatmap,'flag':0},
            'leading_role_heatmap':{'heatmap':leading_role_heatmap,'flag':0},
            'golfclub_heatmap':{'heatmap':golfclub_heatmap,'flag':1},
            'leading_role_keypoints':{'xy':leading_role_keypoints_xy,'cf':leading_role_keypoints_cf,'flag':0},
            'golfclub_keypoints':{'xy':golfclub_keypoints_xy,'cf':golfclub_keypoints_cf,'flag':1},
            'leading_role_bbox':{'xywh':leading_role_bbox_xywh,'cf':leading_role_bbox_cf,'flag':0}
            }
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self,reader,processor,golfer_coco_ratio,dummy_ratio):
        super().__init__()
        self.reader = reader
        self.processor = processor
        self.golfer_coco_ratio = golfer_coco_ratio if isinstance(golfer_coco_ratio,(int,float)) else eval(f'data_schedules.{golfer_coco_ratio}')
        self.dummy_ratio = dummy_ratio if isinstance(dummy_ratio,(int,float)) else eval(f'data_schedules.{dummy_ratio}')
        
    def __getitem__(self,index):
        return self.processor(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)
    
    def __call__(self,epoch_count=None):
        golfer_coco_ratio = self.golfer_coco_ratio if isinstance(self.golfer_coco_ratio,(int,float)) else self.golfer_coco_ratio(epoch_count)
        dummy_ratio = self.dummy_ratio if isinstance(self.dummy_ratio,(int,float)) else self.dummy_ratio(epoch_count)
        self.dataset = self.reader(golfer_coco_ratio,dummy_ratio)
        return self
    
class DataLoader:
    def __init__(self,dataset,batch_size,shuffle,num_workers,pin_memory,prefetch_factor):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
    
    def __call__(self,epoch_count=None):
        if epoch_count is not None:
            dataset = self.dataset(epoch_count)
        else:
            dataset = self.dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor)
        return tqdm(dataloader, position = 0, leave = True, desc=f'EPOCH: {epoch_count}')
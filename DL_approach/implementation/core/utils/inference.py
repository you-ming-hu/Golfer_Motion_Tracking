import core.dataset.common
import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize(
    save_path,
    vis_prob,
    cf_threshold,
    image,
    multi_people_heatmap,multi_people_paf,
    leading_role_heatmap,leading_role_paf,
    golfclub_heatmap,golfclub_paf,
    leading_role_keypoints_xy, leading_role_keypoints_cf,
    golfclub_keypoints_xy, golfclub_keypoints_cf,
    leading_role_bbox_xywh,leading_role_bbox_cf):
    if bool(np.random.binomial(1,p=vis_prob)):
        image_shape = image.shape[:2][::-1]
        
        image = image.copy()
        
        plt.figure(figsize=(30,20))
        plt.subplot(2,2,1)
        
        plt.title('multi_people_heatmap',fontsize=50)
        plt.imshow(image)
        plt.imshow(cv2.resize(np.fmax(multi_people_heatmap.max(axis=-1),multi_people_paf.max(axis=-1)),image_shape),alpha=0.5,cmap='jet')
        plt.axis(False)

        plt.subplot(2,2,2)
        plt.title('leading_role_heatmap',fontsize=50)
        plt.imshow(image)
        plt.imshow(cv2.resize(np.fmax(leading_role_heatmap.max(axis=-1),leading_role_paf.max(axis=-1)),image_shape),alpha=0.5,cmap='jet')
        plt.axis(False)
        
        plt.subplot(2,2,3)
        plt.title('golfclub_heatmap',fontsize=50)
        plt.imshow(image)
        plt.imshow(cv2.resize(np.fmax(golfclub_heatmap.max(axis=-1),golfclub_paf.max(axis=-1)),image_shape),alpha=0.5,cmap='jet')
        plt.axis(False)

        for i in range(len(core.dataset.common.human_keypoints)):
            if leading_role_keypoints_cf[i] > cf_threshold:
                xy = leading_role_keypoints_xy[i]
                cv2.circle(image,xy,5,(1,0,0),2)

        for i in range(len(core.dataset.common.golfclub_keypoints)):
            if golfclub_keypoints_cf[i] > cf_threshold:
                xy = golfclub_keypoints_xy[i]
                cv2.circle(image,xy,5,(0,0,1),2)
                
        if leading_role_bbox_cf > cf_threshold:
            xywh = leading_role_bbox_xywh
            lt = (xywh[:2]-xywh[2:]/2).astype(int)
            rb = (xywh[:2]+xywh[2:]/2).astype(int)
            cv2.rectangle(image,lt,rb,(1,0,0),2)
        
        plt.subplot(2,2,4)
        plt.title('keypoints',fontsize=50)
        plt.imshow(image)
        plt.axis(False)
        
        plt.tight_layout()
        
        image_count = len(list(save_path.iterdir()))
        plt.savefig(save_path.joinpath(f'{image_count:0>5}').with_suffix('.jpg').as_posix(),bbox_inches='tight')
        plt.close()
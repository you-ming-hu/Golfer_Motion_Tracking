from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

uniform_input_image_size = (512,288) #W,H
heatmap_downsample = 4
hm_joint_ratio = 1/15
hm_clubhead_ratio = 1/10

preprocess = A.CLAHE((8,8),(2,2),p=1)
postprocess = A.Sequential([A.ToFloat(p=1),A.Normalize(p=1,max_pixel_value=1),ToTensorV2(p=1)],p=1)

human_keypoints= [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
    ]

human_skeleton = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(5,11),(6,8),(6,12),(7,9),(8,10),(11,12),(11,13),(12,14),(13,15),(14,16)]

golfclub_keypoints = [
    'club_grip',
    'club_head'
    ]
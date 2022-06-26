from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

uniform_input_image_size = (512,288) #W,H
heatmap_downsample = 2

preprocess = A.CLAHE(8,(2,2),p=1)
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

golfclub_keypoints = [
    'club_grip',
    'club_head'
    ]
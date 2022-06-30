from .chained_easy_dict import ChainedEasyDict
import numpy as np
import core.dataset.common
def initialize_config():
    global Config
    Config = ChainedEasyDict('Config')
    with Config:
        Config.AutoGenerate.TrainStages = ['train','val']
    return Config
    
def get_config():
    trial_number = Config.Training.Settings.TrailNumber
    root_seed = Config.Training.Settings.Random.RootSeed
    np.random.seed(root_seed)
    with Config.AutoGenerate as Auto:
        seeds = np.random.randint(0,256,size=(100,4))
        ds_select,ds_aug,ds_shuffle,model_weights = seeds[trial_number]
        Auto.RandomSeed.DataSelection = ds_select
        Auto.RandomSeed.DatasetAugmentation = ds_aug
        Auto.RandomSeed.DatasetShuffle = ds_shuffle
        Auto.RandomSeed.ModelWeight = model_weights
        
        Auto.Common.InputImageSize = core.dataset.common.uniform_input_image_size
        Auto.Common.HeatmapDownsample = core.dataset.common.heatmap_downsample
        Auto.Common.HMJointRatio = core.dataset.common.hm_joint_ratio
        Auto.Common.HMClubheadRatio = core.dataset.common.hm_clubhead_ratio
        Auto.Common.HumanKeypoints = core.dataset.common.human_keypoints
        Auto.Common.GolfclubKeypoints = core.dataset.common.golfclub_keypoints
    return Config
from .chained_easy_dict import ChainedEasyDict
import numpy as np

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
    return Config
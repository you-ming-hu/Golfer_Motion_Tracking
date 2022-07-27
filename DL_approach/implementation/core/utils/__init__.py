import pathlib
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
import numpy as np
import logging

from .configuration import  initialize_config, get_config
from .inference import visualize
import core.dataset.training
import core.lib.losses
import core.lib.schedules.lr_schedules

import core.model

def get_dataloaders(Config):
    coco_path = Config.Training.Dataset.CocoPath
    golfer_path = Config.Training.Dataset.GolferPath
    train_composition = Config.Training.Dataset.Train.Composition
    val_composition = Config.Training.Dataset.Val.Composition
    
    train_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[0],coco_path,golfer_path)
    val_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[1],coco_path,golfer_path,seed=0)
    
    aug_processor = core.dataset.training.DataAugProcessor()
    non_aug_processor = core.dataset.training.DataNonAugProcessor()
    
    train_dataset = core.dataset.training.Dataset(train_reader,aug_processor,**train_composition)
    val_dataset = core.dataset.training.Dataset(val_reader,non_aug_processor,**val_composition)
    
    train_dataloader = core.dataset.training.DataLoader(
        train_dataset,
        Config.Training.Dataset.Train.BatchSize,
        True,
        Config.Training.Dataset.NumWorkers,
        Config.Training.Dataset.PinMemory,
        Config.Training.Dataset.PrefetchFactor
    )
    val_dataloader = core.dataset.training.DataLoader(
        val_dataset,
        Config.Training.Dataset.Val.BatchSize,
        False,
        Config.Training.Dataset.NumWorkers,
        Config.Training.Dataset.PinMemory,
        Config.Training.Dataset.PrefetchFactor
    )
    return {k:v for k,v in zip(Config.AutoGenerate.TrainStages,(train_dataloader, val_dataloader))}

def get_loss_func(Config):
    hybrid_loss = core.lib.losses.HybridLoss(**Config.Training.HybridLoss)
    return hybrid_loss

def get_writers(Config):
    train_writer = SummaryWriter(pathlib.Path(Config.AutoGenerate.SavePath.Logging,'record',Config.AutoGenerate.TrainStages[0]).as_posix())
    val_writer = SummaryWriter(pathlib.Path(Config.AutoGenerate.SavePath.Logging,'record',Config.AutoGenerate.TrainStages[1]).as_posix())
    return {k:v for k,v in zip(Config.AutoGenerate.TrainStages,(train_writer, val_writer))}

def get_loggers(Config):
    train_logger = logging.getLogger(Config.AutoGenerate.TrainStages[0])
    val_logger = logging.getLogger(Config.AutoGenerate.TrainStages[1])

    train_logger.setLevel(logging.INFO)
    val_logger.setLevel(logging.INFO)

    train_logger.addHandler(logging.FileHandler(pathlib.Path(Config.AutoGenerate.SavePath.Logging,Config.AutoGenerate.TrainStages[0]).with_suffix('.txt').as_posix()))
    val_logger.addHandler(logging.FileHandler(pathlib.Path(Config.AutoGenerate.SavePath.Logging,Config.AutoGenerate.TrainStages[1]).with_suffix('.txt').as_posix()))
    
    return {k:v for k,v in zip(Config.AutoGenerate.TrainStages,(train_logger, val_logger))}

def get_model(Config):
    model = core.model.Model(**Config.Model)
    return model

def get_optimizer(model,Config):
    optimizer = getattr(torch.optim,Config.Training.Optimizer.Name)
    return optimizer(model.parameters(),**Config.Training.Optimizer.Param)

def get_lr_scheduler(optimizer,Config):
    lr_scheduler = getattr(core.lib.schedules.lr_schedules,Config.Training.LearningRateSchedule.Name)
    lr_scheduler = lr_scheduler(**Config.Training.LearningRateSchedule.Param)
    return lr_scheduler(optimizer)

def save_config(Config):
    save_root = pathlib.Path(Config.AutoGenerate.SavePath.Logging)
    save_root.mkdir(parents=True,exist_ok=True)
    save_root.joinpath('config.txt').write_text(str(Config))
    pickle.dump(Config,save_root.joinpath('config.pkl').open('wb'))
    SummaryWriter(save_root.joinpath('record','config').as_posix()).add_text('Config',str(Config),0)

def update_stage_result(dataloader,losses):
    dataloader.set_postfix(**{k:'{:.4f}'.format(v) if v is not None else 'None' for k,v in losses.items()})
    
def update_stage_logger(step,steps_per_epoch,logger,losses):
    log_string = []
    log_string.append('-'*20 + f'  {step} / {steps_per_epoch}  ' + '-'*20)
    for k,v in losses.items():
        if v is not None:
            log_string.append(f'{k} = {v:.4f}')
        else:
            log_string.append(f'{k} = None')
    log_string = '\n'.join(log_string)
    logger.info(log_string)
    
def save_model(model,epoch_count,Config):
    save_root = pathlib.Path(Config.AutoGenerate.SavePath.ModelWeights)
    save_root.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),save_root.joinpath(f'{epoch_count:0>3}.pt'))
    
def record_inference(Config,training_epoch_count,image,label,random_state=np.random.RandomState(None)):
    save_path = pathlib.Path(Config.AutoGenerate.SavePath.Logging,'visualize',f'{training_epoch_count:0>3}')
    vis_prob = Config.Record.VisualizeRatio
    threshold = Config.Record.DetectThreshold
    
    image = image.cpu().numpy()
    image = image.transpose([0,2,3,1])
    # image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # image = np.clip(image,0,1)
    
    image_shape = np.array(image.shape[1:3][::-1])
    batch_size = image.shape[0]
    
    label = {s:t.cpu().numpy() if not isinstance(t,dict) else {u:v.cpu().numpy() for u,v in t.items()} for s,t in label.items()}
    
    def get_heatmap(name):
        try:
            heatmap = label[name]['heatmap']
            heatmap = heatmap.transpose([0,2,3,1])
            return heatmap
        except KeyError:
            return [None]*batch_size
    multi_people_heatmap = get_heatmap('multi_people_heatmap')
    leading_role_heatmap = get_heatmap('leading_role_heatmap')
    golfclub_heatmap = get_heatmap('golfclub_heatmap')
    
    def get_paf(name):
        try:
            paf = label[name]['paf']
            paf = paf.transpose([0,2,3,1])
            return paf
        except KeyError:
            return [None]*batch_size
    multi_people_paf = get_paf('multi_people_heatmap')
    leading_role_paf = get_paf('leading_role_heatmap')
    golfclub_paf = get_paf('golfclub_heatmap')
    
    def get_coor_cf(name):
        try:
            xy = np.round(label[name]['xy'] * image_shape).astype(int)
            cf = label[name]['cf']
            return xy,cf
        except KeyError:
            return [None]*batch_size,[None]*batch_size
    leading_role_keypoints_xy,leading_role_keypoints_cf = get_coor_cf('leading_role_keypoints')
    golfclub_keypoints_xy,golfclub_keypoints_cf = get_coor_cf('golfclub_keypoints')
    
    def get_bbox_cf():
        try:
            xywh = np.round(label['leading_role_bbox']['xywh'] * np.tile(image_shape,2)).astype(int)
            cf = label['leading_role_bbox']['cf']
            return xywh,cf
        except KeyError:
            return [None]*batch_size,[None]*batch_size
    leading_role_bbox_xywh,leading_role_bbox_cf = get_bbox_cf()
    
    save_path.mkdir(parents=True,exist_ok=True)
    for data in zip(
        image,
        multi_people_heatmap,multi_people_paf,
        leading_role_heatmap,leading_role_paf,
        golfclub_heatmap,golfclub_paf,
        leading_role_keypoints_xy, leading_role_keypoints_cf,
        golfclub_keypoints_xy, golfclub_keypoints_cf,
        leading_role_bbox_xywh,leading_role_bbox_cf):
        if bool(random_state.binomial(1,p=vis_prob)):
            visualize(save_path, threshold, *data)
    
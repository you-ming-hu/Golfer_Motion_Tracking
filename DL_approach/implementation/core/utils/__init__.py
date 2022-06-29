import pathlib
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pickle
import numpy as np

from .configuration import  initialize_config, get_config
from .inference import visualize
import core.dataset.training
import core.lib.losses
import core.lib.schedules.lr_schedules

import core.models

def get_dataloaders(Config):
    coco_path = Config.Training.Dataset.CocoPath
    golfer_path = Config.Training.Dataset.GolferPath
    base_count = Config.Training.Dataset.Select.BaseCount
    golfer_schedule = Config.Training.Dataset.Select.Golfer
    coco_human_shcedule = Config.Training.Dataset.Select.CocoHuman
    coco_dummy_schedule = Config.Training.Dataset.Select.CocoDummy
    
    train_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[0],coco_path,golfer_path)
    val_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[1],coco_path,golfer_path)
    
    aug_processor = core.dataset.training.DataAugProcessor()
    non_aug_processor = core.dataset.training.DataNonAugProcessor()
    
    train_dataset = core.dataset.training.Dataset(train_reader,aug_processor,base_count,golfer_schedule,coco_human_shcedule,coco_dummy_schedule)
    val_dataset = core.dataset.training.Dataset(val_reader,non_aug_processor,base_count,golfer_schedule,coco_human_shcedule,coco_dummy_schedule)
    
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
    train_writer = SummaryWriter(pathlib.Path(Config.Record.RootPath,'record',Config.AutoGenerate.TrainStages[0]).as_posix())
    val_writer = SummaryWriter(pathlib.Path(Config.Record.RootPath,'record',Config.AutoGenerate.TrainStages[1]).as_posix())
    return {k:v for k,v in zip(Config.AutoGenerate.TrainStages,(train_writer, val_writer))}

def get_model(Config):
    model = getattr(core.models,Config.Model.Name).Model(**Config.Model.Param)
    return model

def get_optimizer(model,Config):
    optimizer = getattr(torch.optim,Config.Training.Optimizer.Name)
    return optimizer(model.parameters(),**Config.Training.Optimizer.Param)

def get_lr_scheduler(optimizer,Config):
    lr_scheduler = getattr(core.lib.schedules.lr_schedules,Config.Training.LearningRateSchedule.Name)
    lr_scheduler = lr_scheduler(**Config.Training.LearningRateSchedule.Param)
    return lr_scheduler(optimizer)

def save_config(Config):
    save_root = pathlib.Path(Config.Record.RootPath)
    save_root.mkdir(parents=True,exist_ok=True)
    save_root.joinpath('config.txt').write_text(str(Config))
    pickle.dump(Config,save_root.joinpath('config.pkl').open('wb'))
    SummaryWriter(save_root.joinpath('record','config').as_posix()).add_text('Config',str(Config),0)

def update_stage_result(dataloader,losses):
    dataloader.set_postfix(**{k:'{:.4f}'.format(v) if v is not None else 'None' for k,v in losses.items()})
    
def save_model(model,epoch_count,Config):
    if Config.Record.SaveModelWeights:
        save_root = pathlib.Path(Config.Record.RootPath,'model_weights')
        save_root.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),save_root.joinpath(f'{epoch_count:0>3}.pt'))
    
def record_inference(Config,training_epoch_count,image,label):
    save_path = pathlib.Path(Config.Record.RootPath,'visualize',f'{training_epoch_count:0>3}')
    vis_prob = Config.Record.VisualizeRatio
    threshold = Config.Record.DetectThreshold
    
    image = image.cpu().numpy()
    image = image.transpose([0,2,3,1])
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image,0,1)
    
    image_shape = np.array(image.shape[1:3][::-1])
    
    label = {s:t.cpu().numpy() if not isinstance(t,dict) else {u:v.cpu().numpy() for u,v in t.items()} for s,t in label.items()}
    
    def get_heatmap(name):
        heatmap = label[name]
        heatmap = heatmap.transpose([0,2,3,1])
        return heatmap
    multi_people_heatmap = get_heatmap('multi_people_heatmap')
    leading_role_heatmap = get_heatmap('leading_role_heatmap')
    golfclub_heatmap = get_heatmap('golfclub_heatmap')
    
    def get_coor_cf(name):
        xy = np.round(label[name]['xy'] * image_shape).astype(int)
        cf = label[name]['cf']
        return xy,cf
    leading_role_keypoints_xy,leading_role_keypoints_cf = get_coor_cf('leading_role_keypoints')
    golfclub_keypoints_xy,golfclub_keypoints_cf = get_coor_cf('golfclub_keypoints')
    
    def get_bbox_cf():
        xywh = np.round(label['leading_role_bbox']['xywh'] * np.tile(image_shape,2)).astype(int)
        cf = label['leading_role_bbox']['cf']
        return xywh,cf
    leading_role_bbox_xywh,leading_role_bbox_cf = get_bbox_cf()
    
    save_path.mkdir(parents=True,exist_ok=True)
    for data in zip(
        image,
        multi_people_heatmap, leading_role_heatmap, golfclub_heatmap,
        leading_role_keypoints_xy, leading_role_keypoints_cf,
        golfclub_keypoints_xy, golfclub_keypoints_cf,
        leading_role_bbox_xywh,leading_role_bbox_cf):
        visualize(save_path, vis_prob, threshold, *data)
    
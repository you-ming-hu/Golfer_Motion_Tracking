import pathlib
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import pickle

from .configuration import  initialize_config, get_config
import core.dataset.training
import core.lib.losses
import core.lib.schedules.lr_schedules

import core.models

def get_dataloaders(Config):
    coco_path = Config.Training.Dataset.CocoPath
    golfer_path = Config.Training.Dataset.GolferPath
    golfer_coco_ratio = Config.Training.Dataset.GolferCocoRatio
    dummy_ratio = Config.Training.Dataset.DummyRatio
    
    train_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[0],coco_path,golfer_path)
    val_reader = core.dataset.training.DataReader(Config.AutoGenerate.TrainStages[1],coco_path,golfer_path)
    
    aug_processor = core.dataset.training.DataAugProcessor()
    non_aug_processor = core.dataset.training.DataNonAugProcessor()
    
    train_dataset = core.dataset.training.Dataset(train_reader,aug_processor,golfer_coco_ratio,dummy_ratio)
    val_dataset = core.dataset.training.Dataset(val_reader,non_aug_processor,golfer_coco_ratio,dummy_ratio)
    
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
    pickle.dump(Config,save_root.joinpath('config.pkl').open('wb'))
    SummaryWriter(save_root.joinpath('record','config').as_posix()).add_text('Config',str(Config),0)

def update_stage_result(dataloader,loss_func):
    contents = {}
    for n,l in loss_func.losses.items():
        if l is not None:
            print(l)
            contents[n] = '{:.4f}'.format(l)
        else:
            contents[n] = 'None'
    dataloader.set_postfix(**contents)
    
def save_model(model,epoch_count,Config):
    if Config.Record.SaveModelWeights:
        save_root = pathlib.Path(Config.Record.RootPath,'model_weights')
        save_root.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),save_root.joinpath(f'{epoch_count:0>3}.pt'))
    
# def record_inference(Config,training_epoch_count,stage,batch_data,output):
#     inf = getattr(inference,Config.Model.Task)
#     save_path = pathlib.Path(Config.Record.RootPath,'results',f'{training_epoch_count:0>3}',stage)
#     inf(save_path,batch_data,output)
    

    
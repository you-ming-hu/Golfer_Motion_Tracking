import random
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

import core.utils

Config = core.utils.get_config()
# create gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cudnn reproducibility
torch.backends.cudnn.deterministic = Config.Training.Settings.Random.cuDNN.Deterministic
torch.backends.cudnn.benchmark = Config.Training.Settings.Random.cuDNN.Benchmark

random.seed(Config.AutoGenerate.RandomSeed.DataSelection)
dataloaders = core.utils.get_dataloaders(Config)

random.seed(Config.AutoGenerate.RandomSeed.DatasetAugmentation) #pixel level augmentation seed
np.random.seed(Config.AutoGenerate.RandomSeed.DatasetAugmentation) #spatial level augmentation seed

loss_func = core.utils.get_loss_func(Config)

torch.manual_seed(Config.AutoGenerate.RandomSeed.ModelWeight)
model = core.utils.get_model(Config)
model = model.to(device)

optimizer = core.utils.get_optimizer(model,Config)
lr_scheduler = core.utils.get_lr_scheduler(optimizer,Config)

writers = core.utils.get_writers(Config)

core.utils.save_config(Config)

stages = Config.AutoGenerate.TrainStages
training_epochs = Config.Training.Settings.Epochs

steps_per_record = Config.Record.Frequence

training_epoch_count = 0
training_step_count = 0
training_progression = 0
training_data_count = 0

amp_scale_train = Config.Training.Settings.AmpScaleTrain
scaler = GradScaler(enabled=amp_scale_train)

torch.manual_seed(Config.AutoGenerate.RandomSeed.DatasetShuffle)
for _ in range(training_epochs):
    #training
    stage = stages[0]
    print('='*50,stage,'='*50)
    dataloader = dataloaders[stage](training_epoch_count)
    progression_per_step = 1 / len(dataloader)
    writer = writers[stage]
    
    model.train()
    for batch_data in dataloader:
        batch_data = {s:t.to(device) if not isinstance(t,dict) else {u:v.to(device) for u,v in t.items()} for s,t in batch_data.items()}
        
        training_progression += progression_per_step
        training_step_count += 1
        training_data_count += batch_data['image'].shape[0]
        
        with autocast(enabled=amp_scale_train,dtype=torch.float32):
            output = model(batch_data['image'])
            hybrid_loss = loss_func(output,batch_data,training_progression)
        
        if hybrid_loss is not None:
            lr = lr_scheduler.step(training_progression)
            
            scaler.scale(hybrid_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
        
        if training_step_count % steps_per_record == 0:
            core.utils.update_stage_result(dataloader,loss_func.result())
            loss_func.log(writer,training_data_count)
            writer.add_scalar('learning_rate',lr,training_data_count)
    
    if training_step_count % steps_per_record != 0:
        loss_func.log(writer,training_data_count)
        writer.add_scalar('learning_rate',lr,training_data_count)
    
    core.utils.save_model(model,training_epoch_count,Config)
                
    #validation
    stage = stages[1]
    print('='*50,stage,'='*50)
    dataloader = dataloaders[stage](training_epoch_count)
    writer = writers[stage]
    
    model.eval()
    for batch_data in dataloader:
        batch_data = {s:t.to(device) if not isinstance(t,dict) else {u:v.to(device) for u,v in t.items()} for s,t in batch_data.items()}
        
        with torch.no_grad():
            with autocast(enabled=amp_scale_train,dtype=torch.float32):
                output = model(batch_data['image'])
                hybrid_loss = loss_func(output,batch_data,training_progression)
        
        core.utils.update_stage_result(dataloader,loss_func.result())
        
        output = model.inference(output)
        core.utils.record_inference(Config,training_epoch_count,batch_data['image'],output)
    
    lr_scheduler.epoch(training_epoch_count,loss_func.losses)
    loss_func.log(writer,training_data_count)

    training_epoch_count+=1
class BaseScheduler:
    def __call__(self,optimizer):
        assert len(optimizer.param_groups) == 1
        self.optimizer = optimizer
        self.init_lr = optimizer.param_groups[0]['lr']
        return self
    
    def step(self,progression):
        lr = self.get_step_lr(progression)
        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr
        return lr
    
    def epoch(self,epoch,losses):
        lr = self.get_epoch_lr(epoch,losses)
        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr
        return lr
    
    def get_step_lr(self,progression):
        raise NotImplementedError
    
    def get_epoch_lr(self,epoch,losses):
        raise NotImplementedError
        
class LinearWarmupExpReduce(BaseScheduler):
    def __init__(self, warmup_epochs, reduce_gamma=-0.5):
        self.warmup_epochs = warmup_epochs
        assert reduce_gamma < 0
        self.reduce_gamma = reduce_gamma
        
    def get_step_lr(self,progression):
        arg1 = (progression/self.warmup_epochs) ** self.reduce_gamma
        arg2 = progression/self.warmup_epochs
        lr = self.init_lr * min(arg1, arg2)
        return lr
    
    def get_epoch_lr(self,epoch,losses):
        return None
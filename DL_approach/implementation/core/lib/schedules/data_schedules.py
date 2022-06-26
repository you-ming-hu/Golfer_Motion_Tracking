class BaseSchedule:
    def __call__(self,epoch_count=None):
        if epoch_count is None:
            return 1
        else:
            return self.call(epoch_count)
    
    def call(self,epoch_count):
        raise NotImplementedError
    
class Constant(BaseSchedule):
    def __init__(self,v):
        self.v = v
    def call(self,epoch_count):
        return self.v
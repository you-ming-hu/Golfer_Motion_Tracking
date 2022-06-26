class BaseSchedule:
    def __call__(self,progression):
        if progression is None:
            return 1
        else:
            return self.call(progression)
    def call(self,progression):
        raise NotImplementedError

class Constant(BaseSchedule):
    def __init__(self,v):
        self.v = v
    def call(self,progression):
        return self.v
    
class WarmUp(BaseSchedule):
    def __init__(self,progression):
        pass
    
    
class Zip(BaseSchedule):
    def __init__(self,):
        pass
    
class LinearDecay(BaseSchedule):
    def __init__(self):
        pass
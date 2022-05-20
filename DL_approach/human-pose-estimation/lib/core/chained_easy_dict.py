import collections
class ChainedEasyDict:
    _editing_mode = False
    
    def __init__(self,name,fathers=[]):
        assert isinstance(name,str)
        fathers = fathers+[name]
        super().__setattr__('_fathers', fathers)
        super().__setattr__('_contents', collections.OrderedDict())
        
    def __repr__(self):
        return f'{__class__.__name__} object: '+'.'.join(self._fathers)
        
    def __str__(self):
        new_line_str = '  \n'
        fathers_str = '.'.join(self._fathers)
        contents_str = new_line_str.join(
            [f'{fathers_str}.{content} = {value}' if not isinstance(value,__class__) else str(value) for content,value in self._contents.items()])
        return contents_str
    
    def __setattr__(self,name,value):
        if __class__._editing_mode:
            assert not name.startswith('_')
            if isinstance(value,str):
                self._contents[name] = f"'{value}'"
            else:
                try:
                    self._contents[name] = value.__name__
                except AttributeError:
                    self._contents[name] = value
                
            super().__setattr__(name, value)
        else:
            raise Exception('can not assign new value outside of editing mode')
    
    def __getattr__(self,name):
        if __class__._editing_mode:
            assert name != 'shape'
            content = __class__(name,self._fathers)
            self._contents[name] = content
            super().__setattr__(name,content)
            return content
        else:
            fathers = super().__getattribute__('_fathers')
            raise AttributeError('.'.join(fathers+[name])+f' is not defined while in editing mode')
    
    def __enter__(self):
        __class__._editing_mode = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        __class__._editing_mode = False
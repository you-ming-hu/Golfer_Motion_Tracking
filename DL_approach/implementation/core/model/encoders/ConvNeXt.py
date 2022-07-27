from .base import BaseEncoder

class Encoder(BaseEncoder):
    def __init__(self,subtype,aux_hog):
        super().__init__('convnext',subtype,aux_hog)
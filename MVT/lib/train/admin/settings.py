from lib.train.admin.environment import env_settings
import torch

class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        # self.use_gpu = False 
        self.use_gpu = torch.cuda.is_available()



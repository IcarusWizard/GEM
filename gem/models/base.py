import torch
from tqdm import tqdm

from gem.utils import select_gpus, step_loader
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    r"""
        Base class of model trainer

        Inputs:

            model : torch.nn.Module
            train_loader : torch.utils.data.DataLoader
            val_loader : torch.utils.data.DataLoader
            test_loader : torch.utils.data.DataLoader
            config : dict, parameters of configuration, i.e. steps, gpu etc.
    """
    def __init__(self, train_loader, val_loader=None, test_loader=None, config={}):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # config gpus
        select_gpus(self.config['gpu']) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # create log writer
        self.writer = SummaryWriter(config['log_name'])

        self.train_iter = step_loader(self.train_loader) # used in training

    def train(self):
        for step in tqdm(range(self.config['steps'])):
            self.train_step()

            if step % self.config['log_step'] == 0:
                self.log_step(step)
        self.log_step(self.config['steps'])

    def train_step(self):
        """
            Perform a single training step 
        """
        raise NotImplementedError()

    def log_step(self, step):
        """
            Perform a single log step 
        """
        raise NotImplementedError()

    def save(self, filename):
        """
            save the current model to file 
        """
        raise NotImplementedError()

    def restore(self, filename):
        """
            restore model from file
        """
        raise NotImplementedError()

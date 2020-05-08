import numpy as np
import os

import torch, torchvision
from ..utils import load_npz

class SequenceDataset(torch.utils.data.Dataset):
    """
        Base dataset of video sequence

        Inputs:

            root : str, path to the dataset
            dataset : str, subset choose from 'train' 'val' 'test' and ''
    """
    def __init__(self, root, dataset=''):
        super().__init__()
        self.root = root
        self.dataset = dataset

        self.datapath = os.path.join(self.root, dataset)

        # load trajlist
        filenames = sorted(os.listdir(self.datapath))
        self.trajlist = [os.path.join(self.datapath, filename) for filename in filenames]

    def __len__(self):
        return len(self.trajlist)

    def process_data(self, value):
        if value.dtype == np.uint8:
            value = (value / 255.0 - 0.5).astype(np.float32) # normalize
        
        if len(value.shape) == 3: # gray image
            value = value[:, np.newaxis]
        elif len(value.shape) == 4: # RGB image
            value = value.transpose((0, 3, 1, 2)) # permute the axis to torch form

        return value

    def __getitem__(self, index):
        traj_file = self.trajlist[index]
        output = load_npz(traj_file)

        return {k : self.process_data(v) for k, v in output.items()}
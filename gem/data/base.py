import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pickle
import PIL
from queue import Queue

import torch, torchvision
from .utils import check_keys
from ..utils import save_npz, load_npz

class ActionSequenceSeparatedDataset(torch.utils.data.Dataset):
    """
        Base dataset of video sequence (Separated Version)

        Inputs:

            root : str, path to the dataset
            dataset : str, subset choose from 'train' 'val' 'test'
            keys: dict, define the kind of data in the dataset
            max_length : int, maxima length of each sequence
            horizon : int, length of each loaded sample (not bigger than max_length)
            fix_start : bool, whether every data start from time step 0, default : False
    """
    def __init__(self, root, dataset, keys={"obs" : ("image_main", 'image'), "action" : ('action', 'txt'), 'reward' : None}, max_length=1000, horizon=1000, fix_start=False):
        super().__init__()
        self.horizon = horizon
        self.root = root
        self.dataset = dataset
        self.max_length = max_length
        self.fix_start = fix_start

        self.keys = keys  

        self.datapath = os.path.join(self.root, dataset)

        # load trajlist
        foldernames = sorted(os.listdir(self.datapath))
        self.trajlist = [os.path.join(self.datapath, foldername) for foldername in foldernames]

        assert self.horizon <= self.max_length, "horizon must smaller than sequence length, i.e. {} <= {}".format(
            self.horizon,
            self.max_length
        )
        
    def load_pkl(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_txt(self, filename):
        return np.loadtxt(filename)

    def load_image(self, filename):
        return plt.imread(filename)

    def __len__(self):
        return len(self.trajlist)

    def __getitem__(self, index):
        # load data
        traj_folder = self.trajlist[index]
        files = os.listdir(traj_folder)

        # set start point
        start = 0 if self.fix_start else random.randint(0, self.max_length - self.horizon)
        end = start + self.horizon

        output = {}

        # load obs
        key, filetype = self.keys['obs']
        if filetype == 'image':
            # first find the suffix
            for filename in files:
                if key in filename:
                    suffix = filename.split('.')[-1]

            prototype = key + '_{}.' + suffix

            images = []
            for i in range(start, start + self.horizon):
                images.append(self.load_image(os.path.join(traj_folder, prototype.format(i))))
            images = np.stack(images)
                
        elif filetype == 'pkl':
            images = self.load_pkl(os.path.join(traj_folder, key + '.pkl'))
        else:
            raise ValueError('obs does not support {} type'.format(filetype))

        if images.dtype == np.uint8:
            images = images / 255.0 # normalize
        
        images = torch.as_tensor(images, dtype=torch.float32)
        if len(images.shape) == 3: # gray image
            images = images.unsqueeze(dim=1).contiguous()
        elif len(images.shape) == 4: # RGB image
            images = images.permute(0, 3, 1, 2).contiguous() # permute the axis to torch form

        output['obs'] = images

        # load action
        if self.keys['action'] is not None:
            key, filetype = self.keys['action']
            if filetype == 'txt':
                data = self.load_txt(os.path.join(traj_folder, key + '.txt'))
            elif filetype == 'pkl':
                data = self.load_pkl(os.path.join(traj_folder, key + '.pkl'))
            else:
                raise ValueError('reward does not support {} type'.format(filetype))            
            data = torch.as_tensor(data, dtype=torch.float32)
            output['action'] = data
        else:
            output['action'] = None

        # load reward
        if self.keys['reward'] is not None:
            key, filetype = self.keys['reward']
            if filetype == 'txt':
                data = self.load_txt(os.path.join(traj_folder, key + '.txt'))
            elif filetype == 'pkl':
                data = self.load_pkl(os.path.join(traj_folder, key + '.pkl'))
            else:
                raise ValueError('reward does not support {} type'.format(filetype))            
            data = torch.as_tensor(data, dtype=torch.float32)
            output['reward'] = data
        
        return {k : v[start:end] for k, v in output.items()}

class ActionSequenceIntegratedDataset(torch.utils.data.Dataset):
    """
        Base dataset of video sequence (Integrated Version)

        Inputs:

            root : str, path to the dataset
            dataset : str, subset choose from 'train' 'val' 'test'
            max_length : int, maxima length of each sequence
            horizon : int, length of each loaded sample (not bigger than max_length)
            fix_start : bool, whether every data start from time step 0, default : False
    """
    def __init__(self, root, dataset, max_length=1000, horizon=1000, fix_start=False):
        super().__init__()
        self.horizon = horizon
        self.root = root
        self.dataset = dataset
        self.max_length = max_length
        self.fix_start = fix_start

        self.keys = keys  

        self.datapath = os.path.join(self.root, dataset)

        # load trajlist
        filenames = sorted(os.listdir(self.datapath))
        self.trajlist = [os.path.join(self.datapath, filename) for filename in filenames]

        assert self.horizon <= self.max_length, "horizon must smaller than sequence length, i.e. {} <= {}".format(
            self.horizon,
            self.max_length
        )

    def __len__(self):
        return len(self.trajlist)

    def __getitem__(self, index):
        # load data
        traj_file = self.trajlist[index]
        output = load_npz(traj_file)

        # set start point
        start = 0 if self.fix_start else random.randint(0, self.max_length - self.horizon)
        end = start + self.horizon

        if 'image' in output.keys():
            images = output['image']
            
            if images.dtype == np.uint8:
                images = images / 255.0 # normalize
            
            images = torch.as_tensor(images, dtype=torch.float32)
            if len(images.shape) == 3: # gray image
                images = images.unsqueeze(dim=1).contiguous()
            elif len(images.shape) == 4: # RGB image
                images = images.permute(0, 3, 1, 2).contiguous() # permute the axis to torch form

            output['image'] = images

        return {k : torch.as_tensor(v[start:end], dtype=torch.float32) for k, v in output.items()}

class ImageDataset(torch.utils.data.Dataset):
    """
        Base image dataset, load everything matches keys in path

        Inputs:

            path : str, path to the dataset
            keys : list[str]
            transform : func[PIL.Image -> tensor]
    """
    def __init__(self, path, keys=['png', 'jpg'], transform=None):
        self.path = path
        self.keys = keys
        self.transform = transform
        
        self.file_list = []

        # search for any matched image in path and its subfolders
        q = Queue()
        q.put(path)
        while not q.empty():
            folder = q.get()
            for file_name in os.listdir(folder):
                if os.path.isdir(os.path.join(folder, file_name)):
                    q.put(os.path.join(folder, file_name)) # add subfolder
                else:
                    if check_keys(file_name, keys):
                        self.file_list.append(os.path.join(folder, file_name))

    def __getitem__(self, index):
        image = PIL.Image.open(self.file_list[index])

        if self.transform is not None:
            image = self.transform(image)

        return image,

    def __len__(self):
        return len(self.file_list)
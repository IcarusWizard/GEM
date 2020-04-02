import torch, torchvision
import numpy as np

import random
import re

class ActionShift(torch.utils.data.Dataset):
    """
        Shift the action sequence by one step
    """
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        data = self._dataset[index]

        assert "action" in data.keys()
        action = data['action']
        zero_action = np.zeros((1, action.shape[1]), dtype=action.dtype)
        data['action'] = np.concatenate([zero_action, action[:-1]], axis=0)

        return data

class Split(torch.utils.data.Dataset):
    """
        split part of the data
    """

    def __init__(self, dataset, horizon, fix_start=False):
        super().__init__()
        self._dataset = dataset
        self.horizon = horizon
        self.fix_start = fix_start

        expr = re.compile(r'(\d+)')

        self.new_traj_indexes = []
        max_length = 0
        for i, traj_file in enumerate(self._dataset.trajlist):
            traj_length = int(expr.findall(traj_file)[-1])
            if traj_length >= horizon:
                self.new_traj_indexes.append(i)
            max_length = max(max_length, traj_length)

        assert len(self.new_traj_indexes) > 0, f"no file match your requirement of horizon {horizon}, the max length is only {max_length}!"

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self.new_traj_indexes)

    def __getitem__(self, index):
        data = self._dataset[self.new_traj_indexes[index]]

        max_length = list(data.values())[0].shape[0]

        # set start point
        start = 0 if self.fix_start else random.randint(0, max_length - self.horizon)
        end = start + self.horizon

        return {k : v[start:end].copy() for k, v in data.items()} # copy will clean the pointer to data, thus the memory can be freed by interpreter

class SeparateImage(torch.utils.data.Dataset):
    """
        split the sequence dataset to separate image dataset
    """
    def __init__(self, dataset, image_per_file=None):
        super().__init__()
        self._dataset = dataset
        self.image_per_file = image_per_file

        if self.image_per_file is not None:
            expr = re.compile(r'(\d+)')

            traj_lengths = [int(expr.findall(f)[-1]) // self.image_per_file for f in self._dataset.trajlist]
            self.length = sum(traj_lengths)

            self.sample_order = []
            for l in traj_lengths:
                order = list(range(l*self.image_per_file))
                random.shuffle(order)
                self.sample_order.append(order)

            self.cumulate_lengths = [0]
            length = 0
            for l in traj_lengths:
                length += l
                self.cumulate_lengths.append(length)
        else:
            self.length = len(self._dataset)

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.image_per_file is not None:
            max_index = len(self.cumulate_lengths) - 1
            min_index = 0
            traj_index = (max_index + min_index) // 2
            while not (self.cumulate_lengths[traj_index] <= index and index < self.cumulate_lengths[traj_index + 1]):
                if self.cumulate_lengths[traj_index] > index: # search left
                    max_index = traj_index
                else: # search right
                    min_index = traj_index
                traj_index = (max_index + min_index) // 2

            index_in_traj = index - self.cumulate_lengths[traj_index]
            
            data = self._dataset[traj_index]
            selected_index = self.sample_order[traj_index][index_in_traj*self.image_per_file:(index_in_traj+1)*self.image_per_file]
            
            return (data['image'][selected_index], )
        else:
            return (self._dataset[index]['image'], )

class KeyMap(torch.utils.data.Dataset):
    """ 
        map old keys to new keys
    """
    def __init__(self, dataset, key_pairs):
        super().__init__()
        self._dataset = dataset
        self.key_pairs = key_pairs

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):        
        data = self._dataset[index]

        for old_key, new_key in self.key_pairs:
            data[new_key] = data.pop(old_key)

        return data

class ToTensor(torch.utils.data.Dataset):
    """
        map the data from ndarray to tensor
    """
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):        
        data = self._dataset[index]

        if isinstance(data, tuple):
            new_data = tuple([torch.as_tensor(d, dtype=torch.float32) for d in data])
        elif isinstance(data, list):
            new_data = [torch.as_tensor(d, dtype=torch.float32) for d in data]
        elif isinstance(data, dict):
            new_data = {k : torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

        return new_data    

class Preload(torch.utils.data.Dataset):
    """
        preload the whole dataset into memory
    """
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset

        print('Preloading ......')
        self.data = []
        for i in range(len(self._dataset)):
            data = self._dataset[i]
            if isinstance(data, dict): # sequence
                self.data.append(data)
            elif isinstance(data, tuple):
                data = data[0]
                for j in range(data.shape[0]):
                    self.data.append(data[j])
        print('Preloading complete!')
        self.dtype = type(self.data[0])

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):    
        if self.dtype == dict:
            return self.data[index]
        else:    
            return (self.data[index], )    

def multiple_wrappers(wrapper_list):
    def wrapper(dataset):
        for wrapper in wrapper_list:
            dataset = wrapper(dataset)
        return dataset
    return wrapper
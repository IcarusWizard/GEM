import torch, torchvision

import random

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
        zero_action = torch.zeros(1, action.shape[1], dtype=action.dtype, device=action.device)
        data['action'] = torch.cat([zero_action, action[:-1]])

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

        data = self._dataset[0]
        self.max_length = list(data.values())[0].shape[0]

        assert self.horizon <= self.max_length, "horizon must smaller than sequence length, i.e. {} <= {}".format(
            self.horizon,
            self.max_length
        )

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        data = self._dataset[index]

        # set start point
        start = 0 if self.fix_start else random.randint(0, self.max_length - self.horizon)
        end = start + self.horizon

        return {k : v[start:end] for k, v in data.items()}

class SeparateImage(torch.utils.data.Dataset):
    """
        split the sequence dataset to separate image dataset
    """
    def __init__(self, dataset, max_length):
        super().__init__()
        self._dataset = dataset
        self.max_length = max_length

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset) * self.max_length

    def __getitem__(self, index):
        traj_index = index // self.max_length
        index_in_traj = index % self.max_length
        
        data = self._dataset[traj_index]

        return (data['image'][index_in_traj], )  # keep the iterable form

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

def multiple_wrappers(wrapper_list):
    def wrapper(dataset):
        for wrapper in wrapper_list:
            dataset = wrapper(dataset)
        return dataset
    return wrapper
import os
import random
from functools import partial
import torch
import torchvision

from .bair_push import load_bair_push, load_bair_push_seq
from .base import SequenceDataset
from .wrapper import ActionShift, SeparateImage, KeyMap, Split, ToTensor, Preload, multiple_wrappers

DATAROOT = './dataset'

def load_sensor_dataset(config, batch_size=None):
    name = config['dataset']
    image_per_file = config['image_per_file']
    preload = config['preload']
    if name == 'bair_push':
        trainset, valset, testset, model_param = load_bair_push(image_per_file=image_per_file)
    else:
        trainset, valset, testset, model_param =  load_env_dataset(name, preload=preload, image_per_file=image_per_file)

    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = config['batch_size'] if preload else config['batch_size'] // config['image_per_file']

    isampler = InfiniteSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=isampler, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    return model_param, train_loader, val_loader, test_loader

def load_env_dataset(name, preload, image_per_file):
    datadir = os.path.join(DATAROOT, name)

    if not os.path.exists(datadir):
        raise ValueError('Please run generate dataset first')

    if preload:
        wrapper = multiple_wrappers([
            partial(SeparateImage, image_per_file=None),
            Preload,
            ToTensor,
        ])         
    else:
        wrapper = multiple_wrappers([
            partial(SeparateImage, image_per_file=image_per_file),
            ToTensor,
        ]) 

    trainset = wrapper(SequenceDataset(os.path.join(datadir, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(datadir, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(datadir, 'test')))

    image = trainset[0][0]
    config = {
        "c" : image.shape[-3],
        "h" : image.shape[-2],
        "w" : image.shape[-1],
    }

    return (trainset, valset, testset, config)

def load_predictor_dataset(config, batch_size=None):
    name = config['dataset']
    horizon = config['horizon']
    fix_start = config['fix_start']
    preload = config['preload']

    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use

    if name == 'bair_push':
        trainset, valset, testset, model_param = load_bair_push_seq(horizon=horizon, fix_start=fix_start)
    else:
        trainset, valset, testset, model_param =  load_env_dataset_seq(name, horizon, fix_start, preload)
        if preload:
            workers = 0 # you dont need multi-loader when preloaded the whole dataset
    
    if batch_size is None:
        batch_size = config['batch_size']

    isampler = InfiniteSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=isampler, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=min(batch_size, len(valset)), num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=min(batch_size, len(testset)), num_workers=workers, pin_memory=True)

    return model_param, train_loader, val_loader, test_loader

def load_env_dataset_seq(name, horizon, fix_start, preload):
    datadir = os.path.join(DATAROOT, name)

    if not os.path.exists(datadir):
        raise ValueError('Please run generate dataset first')
    
    if preload:
        wrapper = multiple_wrappers([
            Preload,
            partial(Split, horizon=horizon, fix_start=fix_start),
            ToTensor,
        ])     
    else:
        wrapper = multiple_wrappers([
            partial(Split, horizon=horizon, fix_start=fix_start),
            ToTensor,
        ]) 

    trainset = wrapper(SequenceDataset(os.path.join(datadir, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(datadir, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(datadir, 'test')))

    data = trainset[0]

    config = {
        "obs" : tuple(data['image'].shape[1:]),
        "action" : data['action'].shape[1],
        'reward' : 1,
    }

    return (trainset, valset, testset, config)
    

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        indexes = []
        while True:
            if len(indexes) == 0:
                indexes = list(range(len(self.data_source)))
                random.shuffle(indexes)
            yield indexes.pop()

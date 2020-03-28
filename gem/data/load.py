import os
from functools import partial
import torch
import torchvision

from .bair_push import load_bair_push, load_bair_push_seq
from .base import SequenceDataset
from .wrapper import ActionShift, SeparateImage, KeyMap, Split, multiple_wrappers

DATAROOT = './dataset'

def load_sensor_dataset(config, batch_size=None):
    name = config['dataset']
    image_per_file = config['image_per_file']
    if name == 'bair_push':
        trainset, valset, testset, model_param = load_bair_push(image_per_file=image_per_file)
    else:
        trainset, valset, testset, model_param =  load_env_dataset(name, image_per_file=image_per_file)

    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = config['batch_size'] // config['image_per_file']

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    filenames = {
        "log_name" : "{}_{}".format(config['model'], config['dataset']),
        "model_name" : "{}_{}".format(config['model'], config['dataset']),
    }

    if config['suffix']:
        for key in filenames.keys():
            filenames[key] += '_{}'.format(config['suffix'])

    return filenames, model_param, train_loader, val_loader, test_loader

def load_env_dataset(name, image_per_file):
    datadir = os.path.join(DATAROOT, name)

    if not os.path.exists(datadir):
        raise ValueError('Please run generate dataset first')

    config = {
        "c" : 3,
        "h" : 64,
        "w" : 64,
    }

    wrapper = partial(SeparateImage, image_per_file=image_per_file)

    trainset = wrapper(SequenceDataset(os.path.join(datadir, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(datadir, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(datadir, 'test')))

    return (trainset, valset, testset, config)

def load_predictor_dataset(config, batch_size=None):
    name = config['dataset']
    horizon = config['horizon']
    fix_start = config['fix_start']

    if name == 'bair_push':
        trainset, valset, testset, model_param = load_bair_push_seq(horizon=horizon, fix_start=fix_start)
    else:
        trainset, valset, testset, model_param =  load_env_dataset_seq(name, horizon, fix_start)

    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = config['batch_size']

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    filenames = {
        "log_name" : "{}_{}".format(config['model'], config['checkpoint']),
        "model_name" : "{}_{}".format(config['model'], config['checkpoint']),
    }

    if config['suffix']:
        for key in filenames.keys():
            filenames[key] += '_{}'.format(config['suffix'])

    return filenames, model_param, train_loader, val_loader, test_loader

def load_env_dataset_seq(name, horizon, fix_start):
    datadir = os.path.join(DATAROOT, name)

    if not os.path.exists(datadir):
        raise ValueError('Please run generate dataset first')

    wrapper = partial(Split, horizon=horizon, fix_start=fix_start)

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
    
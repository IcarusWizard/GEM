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
    if name == 'bair_push':
        trainset, valset, testset, model_param = load_bair_push()
    else:
        trainset, valset, testset, model_param =  load_env_dataset(name, **config)

    workers = min(config['workers'], os.cpu_count()) # compute actual workers in use
    
    if batch_size is None:
        batch_size = config['batch_size']

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

def load_env_dataset(name, **kargs):
    datadir = os.path.join(DATAROOT, name)

    if not os.path.exists(datadir):
        raise ValueError('Please run generate dataset first')

    config = {
        "c" : 3,
        "h" : 64,
        "w" : 64,
    }

    wrapper = multiple_wrappers([
        partial(SeparateImage, max_length=kargs.get('max_length', 500)),
    ])

    trainset = wrapper(SequenceDataset(os.path.join(datadir, 'train')))
    valset = wrapper(SequenceDataset(os.path.join(datadir, 'val')))
    testset = wrapper(SequenceDataset(os.path.join(datadir, 'test')))

    return (trainset, valset, testset, config)
    
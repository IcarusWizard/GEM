import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from multiprocessing import Pool
from google.protobuf.json_format import MessageToDict
import pickle, os, re, time
from functools import partial

from ..utils import load_pkl
from .utils import get_unpack_functions
from .base import ActionSequenceDataset

def load_compressed_dataset(root, horizon, fix_start=False):
    data_folder = os.path.join(root, 'train', os.listdir(os.path.join(root, 'train'))[0]) # choose a folder

    config = {}
    obs = load_pkl(os.path.join(data_folder, 'obs.pkl'))
    max_length = obs.shape[0]

    config['obs'] = obs.shape[1]

    action = load_pkl(os.path.join(data_folder, 'action.pkl'))
    config['action'] = action.shape[1]

    config['reward'] = 1 if 'reward.pkl' in os.listdir(data_folder) else None

    keys = {
        "obs" : ('obs', 'pkl'),
        "action" : ('action', 'pkl'),
        "reward" : ('reward', 'pkl') if config['reward'] else None,
    }

    trainset = ActionSequenceDataset(root, 'train', keys, max_length=max_length, horizon=horizon, fix_start=fix_start)
    valset = ActionSequenceDataset(root, 'val', keys, max_length=max_length, horizon=horizon, fix_start=fix_start)
    testset = ActionSequenceDataset(root, 'test', keys, max_length=max_length, horizon=horizon, fix_start=fix_start)

    return (trainset, valset, testset, config)
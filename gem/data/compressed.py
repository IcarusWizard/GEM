import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from multiprocessing import Pool
from google.protobuf.json_format import MessageToDict
import pickle, os, re, time
from functools import partial

from ..utils import load_npz
from .utils import get_unpack_functions
from .base import ActionSequenceIntegratedDataset

def load_compressed_dataset(root, horizon, fix_start=False):
    data_file = os.path.join(root, 'train', os.listdir(os.path.join(root, 'train'))[0]) # choose a file

    data = load_npz(data_file)
    max_length = data['action'].shape[0]

    config = {
        'obs' : data['image'].shape[1:],
        'emb' : None if 'emb' not in data.keys() else data['emb'].shape[1],
        'action' : data['action'].shape[1],
        'reward' : None if 'reward' not in data.keys() else 1,
    }

    trainset = ActionSequenceIntegratedDataset(root, 'train', max_length=max_length, horizon=horizon, fix_start=fix_start)
    valset = ActionSequenceIntegratedDataset(root, 'val', max_length=max_length, horizon=horizon, fix_start=fix_start)
    testset = ActionSequenceIntegratedDataset(root, 'test', max_length=max_length, horizon=horizon, fix_start=fix_start)

    return (trainset, valset, testset, config)
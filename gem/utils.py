import pickle
import os
import random
import numpy as np
import torch
import argparse

LOG2PI = 0.5 * np.log(2 * np.pi)

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def select_gpus(gpus="0"):
    '''
        gpus -> string, examples: "0", "0,1,2"
    ''' 
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def step_loader(dataloder):
    data = iter(dataloder)
    while True:
        try:
            x = next(data)
        except:
            data = iter(dataloder)
            x = next(data)
        yield x

def nats2bits(nats):
    return nats / np.log(2)

def bits2nats(bits):
    return bits * np.log(2)

def pickle_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_npz(filename, d):
    np.savez_compressed(filename, **d)

def load_npz(filename):
    data = np.load(filename)
    return {k : v for k, v in data.items()}

def save_gif(filename, video, fps=10):
    """
        save the input video to gif
        filename: String
        video: ndarray with shape [T, H, W, C]
    """
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip([video[i]for i in range(video.shape[0])], fps=fps)
    clip.write_gif(filename, verbose=False, logger=None)

def random_move(input_folder, target_folder):
    """randomly move a file from input folder to target folder"""
    filename = random.choice(os.listdir(input_folder))
    os.system(f'mv {os.path.join(input_folder, filename)} {os.path.join(target_folder, filename)}')

def expert_build(max_index):
    """rename expert traj in current path and only maintain max_index number of traj"""
    file_list = sorted(os.listdir('.'))
    file_list.reverse()
    for i, filename in enumerate(file_list):
        if i < max_index:
            target_file = 'expert_traj_{}_501.npz'.format(i)
            os.system(f'mv {filename} {target_file}')
        else:
            os.system(f'rm {filename}')

def get_config_type(v):
    if isinstance(v, list):
        return lambda x : list(map(int, x.split(',')))
    if isinstance(v, bool):
        return lambda x: bool([False, True].index(int(x)))
    return type(v)

def parse_args(config={}):
    parser = argparse.ArgumentParser()
    for k, v in config.items():
        parser.add_argument(f'--{k}', type=get_config_type(v), default=v, help=f'default : {v}')
    args = parser.parse_args()
    return args

def tsplot(ax, data, **kw):
    """plot with std shade"""
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)
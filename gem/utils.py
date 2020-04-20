import pickle
import os
import random
import numpy as np
import torch

LOG2PI = 0.5 * np.log(2 * np.pi)

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

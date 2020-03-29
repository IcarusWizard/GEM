import pickle
import os
import numpy as np

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

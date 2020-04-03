import os
import random
import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.data import load_sensor_dataset
from gem.utils import load_npz, create_dir

SENSOR_DIR = 'checkpoint/sensor'
INPUT_DIR = 'dataset/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoints', nargs='+', type=str)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--step', type=int, default=1)
    args = parser.parse_args()

    num_checkpoints = len(args.checkpoints)

    fig, ax = plt.subplots(num_checkpoints + 1, 8, figsize=(8, num_checkpoints + 1))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    datadir = os.path.join(INPUT_DIR, args.dataset, 'test')
    filename = random.choice(os.listdir(datadir))
    data = load_npz(os.path.join(datadir, filename))
    image = data['image']

    index = list(range(args.start, args.start + args.step * 8, args.step))

    inputs = image[index]
    inputs = torch.as_tensor(inputs / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)

    for j in range(8):
        ax[0, j].imshow(inputs[j].permute(1, 2, 0).cpu().numpy())
        ax[0, j].axis('off')

    for i, checkpoint_name in enumerate(args.checkpoints):
        checkpoint = torch.load(os.path.join(SENSOR_DIR, checkpoint_name + '.pt'), map_location='cpu')
        model = get_sensor_by_checkpoint(checkpoint)
        model.requires_grad_(False)

        codes = model.encode(inputs[[0, -1]])
        step = (codes[1] - codes[0]) / 7.0
        z = torch.stack([codes[0] + i * step for i in range(8)], dim=0)
        interpolation = model.decode(z)

        for j in range(8):
            ax[i+1, j].imshow(interpolation[j].permute(1, 2, 0).cpu().numpy()) 
            ax[i+1, j].axis('off')          

    plt.show()
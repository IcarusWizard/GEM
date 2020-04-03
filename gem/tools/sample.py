import os
import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.utils import load_npz, create_dir

SENSOR_DIR = 'checkpoint/sensor'
INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'outputs/image'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', type=str)
    args = parser.parse_args()
    
    create_dir(OUTPUT_DIR)

    num_checkpoints = len(args.checkpoints)

    fig, ax = plt.subplots(num_checkpoints, 8, figsize=(8, num_checkpoints))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i, checkpoint_name in enumerate(args.checkpoints):
        checkpoint = torch.load(os.path.join(SENSOR_DIR, checkpoint_name + '.pt'), map_location='cpu')
        model = get_sensor_by_checkpoint(checkpoint)
        model.requires_grad_(False)

        sample = model.sample(8)

        for j in range(8):
            ax[i, j].imshow(sample[j].permute(1, 2, 0).cpu().numpy()) 
            ax[i, j].axis('off')          

    plt.show()
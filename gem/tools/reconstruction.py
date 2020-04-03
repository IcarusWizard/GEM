import os
import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.data import load_sensor_dataset
from gem.utils import load_npz, create_dir

SENSOR_DIR = 'checkpoint/sensor'
INPUT_DIR = 'dataset/'
OUTPUT_DIR = 'outputs/image'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoints', nargs='+', type=str)
    args = parser.parse_args()
    
    create_dir(OUTPUT_DIR)

    num_checkpoints = len(args.checkpoints)

    fig, ax = plt.subplots(num_checkpoints + 1, 8, figsize=(8, num_checkpoints + 1))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    config = {
        "dataset" : args.dataset,
        "image_per_file" : 1,
        "preload" : False,
        "workers" : 0,
        "model" : "foo",
        "suffix" : None
    }

    _, _, _, _, test_loader = load_sensor_dataset(config, batch_size=8)
    testset = test_loader.dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)
    inputs = next(iter(test_loader))[0]
    inputs = inputs.squeeze(1)

    for j in range(8):
        ax[0, j].imshow(inputs[j].permute(1, 2, 0).cpu().numpy())
        ax[0, j].axis('off')

    for i, checkpoint_name in enumerate(args.checkpoints):
        checkpoint = torch.load(os.path.join(SENSOR_DIR, checkpoint_name + '.pt'), map_location='cpu')
        model = get_sensor_by_checkpoint(checkpoint)
        model.requires_grad_(False)

        reconstruction = model.decode(model.encode(inputs))

        for j in range(8):
            ax[i+1, j].imshow(reconstruction[j].permute(1, 2, 0).cpu().numpy()) 
            ax[i+1, j].axis('off')          

    plt.show()
import os
import random
import argparse
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.predictor.run_utils import get_predictor_by_checkpoint
from gem.data import load_predictor_dataset
from gem.utils import load_npz, create_dir

PREDICTOR_DIR = 'checkpoint/predictor'
SENSOR_DIR = 'checkpoint/sensor'
OUTPUT_DIR = 'outputs/images'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--checkpoints', nargs='+', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--fix_start', action='store_true')
    args = parser.parse_args()

    output_dir = os.path.join(OUTPUT_DIR, args.output_dir)
    create_dir(output_dir)

    config = {
        "dataset" : args.dataset,
        "horizon" : args.horizon,
        "fix_start" : args.fix_start,
        "preload" : False,
        "workers" : 0,
        "model" : "foo",
        "suffix" : None,
        "checkpoint" : None,
    }

    _, _, _, test_loader = load_predictor_dataset(config, batch_size=8)
    testset = test_loader.dataset
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    batch = next(iter(test_loader))

    images = batch['image'].permute(1, 0, 2, 3, 4)
    T = images.shape[0]
    action = batch['action'].permute(1, 0, 2)

    result = [[images[i, 0]] for i in range(T)]

    for i, checkpoint_name in enumerate(args.checkpoints):
        checkpoint = torch.load(os.path.join(PREDICTOR_DIR, checkpoint_name + '.pt'), map_location='cpu')
        predictor = get_predictor_by_checkpoint(checkpoint)
        predictor.requires_grad_(False)

        sensor_checkpoint_name = "_".join(checkpoint_name.split('_')[1:])
        checkpoint = torch.load(os.path.join(SENSOR_DIR, sensor_checkpoint_name + '.pt'), map_location='cpu')
        sensor = get_sensor_by_checkpoint(checkpoint)
        sensor.requires_grad_(False)

        _images = images.squeeze(dim=1)
        code = sensor.encode(_images)
        code = code.unsqueeze(dim=1)

        generation = predictor.generate(code[0], code.shape[0], action)
        generated_code = generation['obs']
        generated_video = sensor.decode(generated_code.view(-1, code.shape[-1]))
        generated_video = generated_video.view(*code.shape[:2], *generated_video.shape[1:])
        diff = (generated_video - images + 1) / 2

        for i in range(T):
            result[i].append(generated_video[i, 0])
            result[i].append(diff[i, 0])

    result = [torch.clamp(torch.cat(r, dim=1), 0, 1).permute(1, 2, 0).numpy() for r in result]

    for i in range(T):
        plt.imsave(os.path.join(output_dir, f'{i}.png'), result[i])
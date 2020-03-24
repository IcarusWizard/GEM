import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.utils import pickle_data, create_dir

import os, argparse

import degmo
from gem.models.sensor.run_utils import get_model_by_checkpoint
from degmo.utils import select_gpus

from gem.data import make_dataset

from gem.utils import save_npz

LOGDIR = os.path.join('logs', 'sensor')
MODELDIR = os.path.join('checkpoint', 'sensor')
DATADIR = './dataset'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    select_gpus(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(os.path.join(MODELDIR, args.checkpoint + '.pt'), map_location='cpu')
    config = checkpoint['config']

    model = get_model_by_checkpoint(checkpoint)
    model.requires_grad_(False)
    model = model.to(device)

    OUTPUT_ROOT = os.path.join(DATADIR, args.checkpoint + '_compress')
    create_dir(OUTPUT_ROOT)

    # config dataset
    trainset, valset, testset, dataset_config = make_dataset(config['dataset'] + '_seq')

    sets = {
        'train' : trainset,
        'val' : valset,
        'test' : testset,
    }

    for name, dataset in sets.items():
        dataset_folder = os.path.join(OUTPUT_ROOT, name)
        create_dir(dataset_folder)
        
        print('In {} set'.format(name))

        for i in tqdm(range(len(dataset))):
            traj_file = 'traj_{}.npz'.format(i)

            data = dataset[i]
            output = {}

            obs = data['image']
            output['image'] = (obs.numpy() * 255).as_type(np.uint8)
            obs = obs.to(device)
            z = model.encode(obs)
            output['emb'] = z.cpu().numpy()

            # copy every others
            for k in data.keys():
                if not (k == 'image' or k == 'emb'):
                    output[k] = data[k].numpy()

            save_npz(traj_file, output)
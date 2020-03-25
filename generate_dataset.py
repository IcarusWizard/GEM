import numpy as np
from tqdm import tqdm

from gem.utils import create_dir
from gem.envs.utils import make_env, random_generate

import os, argparse

DATADIR = './dataset'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='dmc_walker_walk')

    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=1000)

    parser.add_argument('--train_traj_num', type=int, default=200)
    parser.add_argument('--test_traj_num', type=int, default=5)

    parser.add_argument()

    args = parser.parse_args()

    config = args.__dict__ 

    datadir = os.path.join(DATADIR, config['env'])
    
    config['datadir'] = os.path.join(datadir, 'train')
    create_dir(config['datadir'])
    random_generate(make_env, config['train_traj_num'])

    config['datadir'] = os.path.join(datadir, 'val')
    create_dir(config['datadir'])
    random_generate(make_env, config['test_traj_num'])

    config['datadir'] = os.path.join(datadir, 'test')
    create_dir(config['datadir'])
    random_generate(make_env, config['test_traj_num'])
    
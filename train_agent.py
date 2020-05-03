import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

import os, argparse

from gem.utils import create_dir
from gem.agents.run_utils import config_agent
from gem.utils import setup_seed

from gem.data import load_sensor_dataset
from gem.agents.config import get_default_agent_config
from gem.envs.utils import make_imagine_env, make_env

LOGDIR = os.path.join('logs', 'agent')
SENSORDIR = os.path.join('checkpoint', 'sensor')
PREDICTORDIR = os.path.join('checkpoint', 'predictor')
AGENTDIR = os.path.join('checkpoint', 'agent')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    default_config = get_default_agent_config()
    for k, v in default_config.items():
        parser.add_argument(f'--{k}', type=type(v), default=v, help=f'default : {v}')
    args = parser.parse_args()

    config = args.__dict__

    create_dir(AGENTDIR)

    # setup random seed
    seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))
    
    # get environment
    real_env = make_env(config)
    world_model = make_imagine_env(config['checkpoint'])
    config['state_dim'] = world_model.state_dim
    config['action_dim'] = world_model.action_dim

    # config dataset
    _, _, observation_loader, _, _ = load_sensor_dataset(config)

    agent, model_param, filename = config_agent(config)

    config['model_param'] = model_param
    config['log_name'] = os.path.join(LOGDIR, filename)

    trainer_class = agent.get_trainer()

    trainer = trainer_class(agent, world_model, real_env, observation_loader, config)
    trainer.train()
    trainer.save(os.path.join(AGENTDIR, '{}.pt'.format(filename)))
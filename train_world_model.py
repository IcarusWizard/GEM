import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.models.mix.trainer import MixTrainer
from gem.models.mix.run_utils import config_world_model
from gem.models.mix.config import get_default_world_model_config, ModelLogDir, ModelDir
from gem.utils import setup_seed, create_dir, parse_args

from gem.data import load_predictor_dataset

if __name__ == '__main__':
    args = parse_args(get_default_world_model_config())

    config = args.__dict__

    create_dir(ModelDir)

    # setup random seed
    seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))

    # config dataset
    dataset_config, train_loader, val_loader, test_loader = load_predictor_dataset(config)
    sample_data = train_loader.dataset[0]
    config['predict_reward'] = 'reward' in sample_data.keys()
    config['action_dim'] = dataset_config['action']

    sensor, predictor, sensor_param, predictor_param, filename = config_world_model(config, dataset_config)

    config['sensor_param'] = sensor_param
    config['predictor_param'] = predictor_param
    config['log_name'] = os.path.join(ModelLogDir, filename)

    trainer = MixTrainer(sensor, predictor, train_loader, val_loader, test_loader, config)
    trainer.train()
    trainer.save(os.path.join(ModelDir, '{}.pt'.format(filename)))
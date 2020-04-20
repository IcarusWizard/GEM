import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm
import importlib

import os, argparse

from gem.utils import create_dir
from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.predictor.run_utils import config_predictor
from gem.utils import setup_seed

from gem.data import load_predictor_dataset

LOGDIR = os.path.join('logs', 'predictor')
SENSORDIR = os.path.join('checkpoint', 'sensor')
PREDICTORDIR = os.path.join('checkpoint', 'predictor')
DATADIR = './dataset'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RSSM')
    parser.add_argument('--checkpoint', type=str)

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--hidden_dim', type=int, default=512)
    model_parser.add_argument('--stoch_dim', type=int, default=32)
    model_parser.add_argument('--decoder_hidden_layers', type=int, default=2)
    model_parser.add_argument('--decoder_features', type=int, default=512)
    model_parser.add_argument('--decoder_activation', type=str, default='elu')
    model_parser.add_argument('--action_mimic', type=bool, default=False)

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=64)
    train_parser.add_argument('--horizon', type=int, default=50)
    train_parser.add_argument('--fix_start', action='store_true')
    train_parser.add_argument('--preload', action='store_true')
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=10000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--beta1', type=float, default=0.9)
    train_parser.add_argument('--beta2', type=float, default=0.999)

    log_parser = parser.add_argument_group('log', "parameters for log config")
    log_parser.add_argument('--log_step', type=int, default=500, help='log period')
    log_parser.add_argument('--fps', type=int, default=15)
    log_parser.add_argument('--suffix', type=str, default=None, help='suffix in log folder and model file')

    args = parser.parse_args()

    config = args.__dict__

    create_dir(PREDICTORDIR)

    # setup random seed
    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))
    
    checkpoint = torch.load(os.path.join(SENSORDIR, args.checkpoint + '.pt'), map_location='cpu')

    coder = get_sensor_by_checkpoint(checkpoint)
    coder.requires_grad_(False)

    config['dataset'] = checkpoint['config']['dataset']

    # config dataset
    filenames, dataset_config, train_loader, val_loader, test_loader = load_predictor_dataset(config)
    sample_data = train_loader.dataset[0]
    config['predict_reward'] = 'reward' in sample_data.keys()
    config['latent_dim'] = coder.latent_dim
    config['action_dim'] = dataset_config['action']

    model, model_param = config_predictor(config)

    config['model_param'] = model_param
    config['log_name'] = os.path.join(LOGDIR, '{}'.format(filenames['log_name']))

    trainer_class = model.get_trainer()

    trainer = trainer_class(model, coder, train_loader, val_loader, test_loader, config)
    trainer.train()
    trainer.save(os.path.join(PREDICTORDIR, '{}.pt'.format(filenames['model_name'])))
    
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.utils import pickle_data, create_dir

import os, argparse

import degmo, gem
from gem.models.sensor.run_utils import get_model_by_checkpoint
from degmo.utils import setup_seed, nats2bits, config_dataset

from gem.data import make_dataset
from gem.models.prediction import GRUBaseline

LOGDIR = os.path.join('logs', 'predictor')
SENSORDIR = os.path.join('checkpoint', 'sensor')
PREDICTORDIR = os.path.join('checkpoint', 'predictor')
DATADIR = './dataset'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GRUBaseline')
    parser.add_argument('--dataset', type=str, default='compressed')
    parser.add_argument('--checkpoint', type=str)

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--hidden_dim', type=int, default=512)
    model_parser.add_argument('--network_type', type=str, default='conv')
    model_parser.add_argument('--decoder_hidden_layers', type=int, default=5)
    model_parser.add_argument('--decoder_features', type=int, default=512)
    model_parser.add_argument('--action_mimic', type=bool, default=True)
    model_parser.add_argument('--predict_reward', type=bool, default=False)

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--horizon', type=int, default=30)
    train_parser.add_argument('--fix_start', action='store_true')
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=10000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--beta1', type=float, default=0.9)
    train_parser.add_argument('--beta2', type=float, default=0.999)

    log_parser = parser.add_argument_group('log', "parameters for log config")
    log_parser.add_argument('--log_step', type=int, default=500, help='log period')
    log_parser.add_argument('--suffix', type=str, default=None, help='suffix in log folder and model file')

    args = parser.parse_args()

    config = args.__dict__

    create_dir(PREDICTORDIR)

    # setup random seed
    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))
    
    checkpoint = torch.load(os.path.join(SENSORDIR, args.checkpoint + '.pt'))
    config = checkpoint['config']

    coder = get_model_by_checkpoint(checkpoint)
    coder.requires_grad_(False)

    OUTPUT_ROOT = os.path.join(DATADIR, args.checkpoint + '_compress')

    # config dataset
    filenames, dataset_config, train_loader, val_loader, test_loader = config_dataset(config, root=OUTPUT_ROOT, horizon=args.horizon, fix_start=args.fix_start)

    model_param = {
        'obs_dim' : dataset_config['obs'],
        "action_dim" : dataset_config['action'],
        "action_mimic" : config['action_mimic'],
        "predcit_reward" : config['predict_reward'],
        "decoder_config" : {
            "hidden_layers" : config['decoder_hidden_layers'],
            "features" : config['decoder_features'],
            "activation" : torch.nn.ELU
        }
    }

    config['model_param'] = model_param
    config['log_name'] = os.path.join(LOGDIR, '{}'.format(filenames['log_name']))

    model_class = getattr(gem.models.prediction, config['model'])
    model = model_class(**model_param)
    trainer_class = model.get_trainer()

    trainer = trainer_class(model, coder, train_loader, val_loader, test_loader, config)
    trainer.train()
    trainer.save(os.path.join(PREDICTORDIR, '{}.pt'.format(filenames['model_name'])))
    
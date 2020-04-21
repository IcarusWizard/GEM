import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

import os, argparse

from gem.models.sensor.run_utils import config_sensor
from gem.data import load_sensor_dataset
from gem.utils import setup_seed

LOGDIR = os.path.join('logs', 'sensor')
MODELDIR = os.path.join('checkpoint', 'sensor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bair_push')
    parser.add_argument('--model', type=str, default='VAE', help='choose from VAE, FVAE')

    model_parser = parser.add_argument_group('model', 'parameters for model config')
    model_parser.add_argument('--latent_dim', type=int, default=32)
    model_parser.add_argument('--free_nats', type=float, default=20)
    model_parser.add_argument('--output_type', type=str, default='gauss',
                              help='the output mode for vae decoder, choose from fix_std, gauss, bernoulli')
    model_parser.add_argument('--network_type', type=str, default='conv')
    model_parser.add_argument('--hidden_layers', type=int, default=0)
    model_parser.add_argument('--features', type=int, default=256)
    model_parser.add_argument('--conv_features', type=int, default=256)
    model_parser.add_argument('--down_sampling', type=int, default=3)
    model_parser.add_argument('--res_layers', nargs='+', type=int, default=[0])
    model_parser.add_argument('--use_batchnorm', action='store_true')
    model_parser.add_argument('--flow_features', type=int, default=1024)
    model_parser.add_argument('--flow_hidden_layers', type=int, default=4)
    model_parser.add_argument('--flow_num_transformation', type=int, default=8)
    model_parser.add_argument('--n_critic', type=int, default=1)

    train_parser = parser.add_argument_group('training', "parameters for training config")
    train_parser.add_argument('--seed', type=int, default=None, help='manuall random seed')
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--image_per_file', type=int, default=2)
    train_parser.add_argument('--preload', action='store_true')
    train_parser.add_argument('--gpu', type=str, default='0')
    train_parser.add_argument('--workers', type=int, default=9999,
                              help='how many workers use for dataloader, default is ALL')
    train_parser.add_argument('--steps', type=int, default=100000)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    train_parser.add_argument('--beta1', type=float, default=0.9)
    train_parser.add_argument('--beta2', type=float, default=0.999)

    log_parser = parser.add_argument_group('log', "parameters for log config")
    log_parser.add_argument('--log_step', type=int, default=1000, help='log period')
    log_parser.add_argument('--suffix', type=str, default=None, help='suffix in log folder and model file')

    args = parser.parse_args()
    
    config = args.__dict__
    
    # setup random seed
    seed = args.seed if args.seed else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))

    # create output folder
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(MODELDIR):
        os.makedirs(MODELDIR)

    # config dataset
    filenames, model_param, train_loader, val_loader, test_loader = load_sensor_dataset(config)

    # config model
    model, model_param = config_sensor(config, model_param)

    config['model_param'] = model_param
    config['log_name'] = os.path.join(LOGDIR, '{}'.format(filenames['log_name']))

    # train the model
    trainer = model.get_trainer()
    trainer = trainer(model, train_loader, val_loader, test_loader, config)
    trainer.train()

    # save final checkpoint
    trainer.save(os.path.join(MODELDIR, '{}.pt'.format(filenames['model_name'])))
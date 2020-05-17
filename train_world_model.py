import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.models.mix.trainer import MixTrainer
from gem.models.mix.run_utils import config_world_model, get_world_model_by_checkpoint
from gem.models.mix.config import get_default_world_model_config, ModelLogDir, ModelDir
from gem.utils import setup_seed, create_dir, parse_args

from gem.data import load_predictor_dataset

if __name__ == '__main__':
    args = parse_args(get_default_world_model_config())

    config = args.__dict__

    if len(config['model_checkpoint']) > 0: # resume training

        # load checkpoint
        checkpoint = torch.load(os.path.join(ModelDir, config['model_checkpoint'] + '.pt'), map_location='cpu')
        sensor, predictor = get_world_model_by_checkpoint(checkpoint, eval_mode=False)

        last_config = checkpoint['config']
        last_config['start_step'] = last_config['steps']
        last_config['steps'] += config['steps']
        last_config['model_checkpoint'] = config['model_checkpoint']

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        last_config['seed'] = seed
        print('using seed {}'.format(seed))

        config = last_config

        # config dataset
        _, train_loader, val_loader, test_loader = load_predictor_dataset(config)

        # train the model
        trainer = MixTrainer(sensor, predictor, train_loader, val_loader, test_loader, config)
        trainer.restore(checkpoint)
        trainer.train()

        # save final checkpoint
        os.system('mv {} {}'.format(
            os.path.join(ModelDir, '{}.pt'.format(config['model_checkpoint'])),
            os.path.join(ModelDir, '{}.pt.old'.format(config['model_checkpoint']))
        ))
        trainer.save(os.path.join(ModelDir, '{}.pt'.format(config['model_checkpoint'])))

    else: # train from scratch
        
        config['start_step'] = 0

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
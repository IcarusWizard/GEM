import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.sensor.config import SensorDir
from gem.models.predictor.run_utils import config_predictor, get_predictor_by_checkpoint
from gem.models.predictor.config import get_default_predictor_config, PredictorLogDir, PredictorDir
from gem.utils import setup_seed, create_dir, parse_args

from gem.data import load_predictor_dataset

if __name__ == '__main__':
    args = parse_args(get_default_predictor_config())

    config = args.__dict__

    if len(config['predictor_checkpoint']) > 0: # resume training

        # load checkpoint
        checkpoint = torch.load(os.path.join(PredictorDir, config['predictor_checkpoint'] + '.pt'), map_location='cpu')
        predictor = get_predictor_by_checkpoint(checkpoint, eval_mode=False)

        last_config = checkpoint['config']
        last_config['start_step'] = last_config['steps']
        last_config['steps'] += config['steps']
        last_config['predictor_checkpoint'] = config['predictor_checkpoint']
        last_config['gpu'] = config['gpu']

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        last_config['seed'] = seed
        print('using seed {}'.format(seed))

        config = last_config

        # config sensor
        sensor_checkpoint = torch.load(os.path.join(SensorDir, args.sensor_checkpoint + '.pt'), map_location='cpu')

        sensor = get_sensor_by_checkpoint(sensor_checkpoint)
        sensor.requires_grad_(False)

        # config dataset
        _, train_loader, val_loader, test_loader = load_predictor_dataset(config)

        # train the model
        trainer = predictor.get_trainer()
        trainer = trainer(predictor, sensor, train_loader, val_loader, test_loader, config)
        trainer.restore(checkpoint)
        trainer.train()

        # save final checkpoint
        os.system('mv {} {}'.format(
            os.path.join(PredictorDir, '{}.pt'.format(config['predictor_checkpoint'])),
            os.path.join(PredictorDir, '{}.pt.old'.format(config['predictor_checkpoint']))
        ))
        trainer.save(os.path.join(PredictorDir, '{}.pt'.format(config['predictor_checkpoint'])))

    else: # train from scratch
        
        config['start_step'] = 0

        create_dir(PredictorDir)

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        config['seed'] = seed
        print('using seed {}'.format(seed))
        
        checkpoint = torch.load(os.path.join(SensorDir, args.sensor_checkpoint + '.pt'), map_location='cpu')

        sensor = get_sensor_by_checkpoint(checkpoint)
        sensor.requires_grad_(False)

        config['dataset'] = checkpoint['config']['dataset']

        # config dataset
        dataset_config, train_loader, val_loader, test_loader = load_predictor_dataset(config)
        sample_data = train_loader.dataset[0]
        config['predict_reward'] = 'reward' in sample_data.keys()
        config['latent_dim'] = sensor.latent_dim
        config['action_dim'] = dataset_config['action']

        predictor, predictor_param, filename = config_predictor(config)

        config['predictor_param'] = predictor_param
        config['log_name'] = os.path.join(PredictorLogDir, filename)

        trainer_class = predictor.get_trainer()

        trainer = trainer_class(predictor, sensor, train_loader, val_loader, test_loader, config)
        trainer.train()
        trainer.save(os.path.join(PredictorDir, '{}.pt'.format(filename)))
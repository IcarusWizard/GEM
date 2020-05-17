import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

from gem.models.sensor.config import get_default_sensor_config, SensorLogDir, SensorDir
from gem.models.sensor.run_utils import config_sensor, get_sensor_by_checkpoint
from gem.data import load_sensor_dataset
from gem.utils import create_dir, setup_seed, parse_args

if __name__ == '__main__':
    args = parse_args(get_default_sensor_config())
    
    config = args.__dict__

    if len(config['sensor_checkpoint']) > 0: # resume training

        # load checkpoint
        checkpoint = torch.load(os.path.join(SensorDir, config['sensor_checkpoint'] + '.pt'), map_location='cpu')
        sensor = get_sensor_by_checkpoint(checkpoint, eval_mode=False)

        last_config = checkpoint['config']
        last_config['start_step'] = last_config['steps']
        last_config['steps'] += config['steps']
        last_config['sensor_checkpoint'] = config['sensor_checkpoint']

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        last_config['seed'] = seed
        print('using seed {}'.format(seed))

        config = last_config

        # config dataset
        _, train_loader, val_loader, test_loader = load_sensor_dataset(config)

        # train the model
        trainer = sensor.get_trainer()
        trainer = trainer(sensor, train_loader, val_loader, test_loader, config)
        trainer.restore(checkpoint)
        trainer.train()

        # save final checkpoint
        os.system('mv {} {}'.format(
            os.path.join(SensorDir, '{}.pt'.format(config['sensor_checkpoint'])),
            os.path.join(SensorDir, '{}.pt.old'.format(config['sensor_checkpoint']))
        ))
        trainer.save(os.path.join(SensorDir, '{}.pt'.format(config['sensor_checkpoint'])))

    else: # train from scratch
        
        config['start_step'] = 0

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        config['seed'] = seed
        print('using seed {}'.format(seed))

        # create output folder
        create_dir(SensorLogDir)
        create_dir(SensorDir)

        # config dataset
        sensor_param, train_loader, val_loader, test_loader = load_sensor_dataset(config)

        # config model
        sensor, sensor_param, filename = config_sensor(config, sensor_param)

        config['sensor_param'] = sensor_param
        config['log_name'] = os.path.join(SensorLogDir, filename)

        # train the model
        trainer = sensor.get_trainer()
        trainer = trainer(sensor, train_loader, val_loader, test_loader, config)
        trainer.train()

        # save final checkpoint
        trainer.save(os.path.join(SensorDir, '{}.pt'.format(filename)))
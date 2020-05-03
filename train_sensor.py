import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision

from gem.models.sensor.config import get_default_sensor_config, SensorLogDir, SensorDir
from gem.models.sensor.run_utils import config_sensor
from gem.data import load_sensor_dataset
from gem.utils import create_dir, setup_seed, parse_args

if __name__ == '__main__':
    args = parse_args(get_default_sensor_config())
    
    config = args.__dict__
    
    # setup random seed
    seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))

    # create output folder
    create_dir(SensorLogDir)
    create_dir(SensorDir)

    # config dataset
    model_param, train_loader, val_loader, test_loader = load_sensor_dataset(config)

    # config model
    model, model_param, filename = config_sensor(config, model_param)

    config['model_param'] = model_param
    config['log_name'] = os.path.join(SensorLogDir, filename)

    # train the model
    trainer = model.get_trainer()
    trainer = trainer(model, train_loader, val_loader, test_loader, config)
    trainer.train()

    # save final checkpoint
    trainer.save(os.path.join(SensorDir, '{}.pt'.format(filename)))
import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.sensor.config import SensorDir
from gem.models.predictor.run_utils import config_predictor
from gem.models.predictor.config import get_default_predictor_config, PredictorLogDir, PredictortDir
from gem.utils import setup_seed, create_dir, parse_args

from gem.data import load_predictor_dataset

if __name__ == '__main__':
    args = parse_args(get_default_predictor_config())

    config = args.__dict__

    create_dir(PredictortDir)

    # setup random seed
    seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))
    
    checkpoint = torch.load(os.path.join(SensorDir, args.checkpoint + '.pt'), map_location='cpu')

    sensor = get_sensor_by_checkpoint(checkpoint)
    sensor.requires_grad_(False)

    config['dataset'] = checkpoint['config']['dataset']

    # config dataset
    dataset_config, train_loader, val_loader, test_loader = load_predictor_dataset(config)
    sample_data = train_loader.dataset[0]
    config['predict_reward'] = 'reward' in sample_data.keys()
    config['latent_dim'] = sensor.latent_dim
    config['action_dim'] = dataset_config['action']

    model, model_param, filename = config_predictor(config)

    config['model_param'] = model_param
    config['log_name'] = os.path.join(PredictorLogDir, filename)

    trainer_class = model.get_trainer()

    trainer = trainer_class(model, sensor, train_loader, val_loader, test_loader, config)
    trainer.train()
    trainer.save(os.path.join(PredictortDir, '{}.pt'.format(filename)))
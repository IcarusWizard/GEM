import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm
import argparse

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.sensor.config import SensorDir
from gem.models.predictor.run_utils import get_predictor_by_checkpoint
from gem.models.predictor.config import get_default_predictor_config, PredictorDir
from gem.models.mix.run_utils import get_world_model_by_checkpoint
from gem.models.mix.config import ModelDir
from gem.utils import step_loader, select_gpus, tsplot

from gem.data import load_predictor_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default='')
    parser.add_argument('--predictor_checkpoint', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batchs', type=int, default=100)
    args = parser.parse_args()

    if len(args.model_checkpoint) > 0:
        model_checkpoint = torch.load(os.path.join(ModelDir, args.model_checkpoint + '.pt'), map_location='cpu')
        config = model_checkpoint['config']
        sensor, predictor = get_world_model_by_checkpoint(model_checkpoint)
    elif len(args.predictor_checkpoint) > 0:
        predictor_checkpoint = torch.load(os.path.join(PredictorDir, args.predictor_checkpoint + '.pt'), map_location='cpu')
        config = predictor_checkpoint['config']
        predictor = get_predictor_by_checkpoint(predictor_checkpoint)
        sensor_checkpoint = torch.load(os.path.join(SensorDir, config['sensor_checkpoint'] + '.pt'), map_location='cpu')
        sensor = get_sensor_by_checkpoint(sensor_checkpoint)
    else:
        raise ValueError('at least one checkpoint should be given')
    
    assert config['predictor'] == 'RSSM', 'currently only RSSM support kl test'
    predictor.requires_grad_(False)
    sensor.requires_grad_(False)

    # config dataset
    config['preload'] = False
    config['workers'] = 4
    _, data_loader, _, _ = load_predictor_dataset(config)
    data_iter = step_loader(data_loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sensor = sensor.to(device)
    predictor = predictor.to(device)

    kls = []
    for i in tqdm(range(args.batchs)):
        batch = next(data_iter)
        obs = batch['image'].permute(1, 0, 2, 3, 4).to(device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(device).contiguous() if config['predict_reward'] else None  

        T, B = obs.shape[:2]
        obs = obs.view(T * B, *obs.shape[2:])
        emb = sensor.encode(obs, output_dist=True).mode().view(T, B, -1)

        predictor_loss, prediction, info = predictor(emb, action, reward, use_emb_loss=False)

        kls.append(torch.sum(prediction['kl'], dim=-1).cpu())
    
    kls = torch.cat(kls, dim=1)
    kls = kls.numpy().T

    fig, ax = plt.subplots()
    tsplot(ax, kls)
    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm
import argparse
from functools import partial

from gem.models.sensor.run_utils import get_sensor_by_checkpoint
from gem.models.sensor.config import SensorDir
from gem.models.predictor.run_utils import get_predictor_by_checkpoint
from gem.models.predictor.config import get_default_predictor_config, PredictorDir
from gem.models.mix.run_utils import get_world_model_by_checkpoint
from gem.models.mix.config import ModelDir
from gem.serial.run_utils import get_serial_agent_by_checkpoint
from gem.serial.config import SerialDir
from gem.utils import step_loader, select_gpus, tsplot, create_dir

from gem.data import load_predictor_dataset
from gem.data.wrapper import multiple_wrappers, Split, ToTensor
from gem.data.base import SequenceDataset
from gem.data.load import InfiniteSampler

LOG2PI = np.log(np.pi * 2) * 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial_agent_checkpoint', type=str, default='')
    parser.add_argument('--model_checkpoint', type=str, default='')
    parser.add_argument('--predictor_checkpoint', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batchs', type=int, default=100)
    args = parser.parse_args()

    output_folder = 'outputs/predictor/prediction'
    create_dir(output_folder)

    if len(args.serial_agent_checkpoint) > 0:
        model_checkpoint = torch.load(os.path.join(SerialDir, args.serial_agent_checkpoint + '.pt'), map_location='cpu')
        config = model_checkpoint['config']
        sensor, predictor, controller = get_serial_agent_by_checkpoint(model_checkpoint)
        datafolder = os.path.join(config['log_name'], 'trajs')

        wrapper = multiple_wrappers([
            partial(Split, horizon=100, fix_start=False),
            ToTensor,
        ]) 

        dataset = wrapper(SequenceDataset(datafolder))
        isampler = InfiniteSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], sampler=isampler, 
                                                  num_workers=os.cpu_count(), pin_memory=True)

        filename = 'serial_agent_' + args.serial_agent_checkpoint + '_{}.txt'
    elif len(args.model_checkpoint) > 0:
        model_checkpoint = torch.load(os.path.join(ModelDir, args.model_checkpoint + '.pt'), map_location='cpu')
        config = model_checkpoint['config']
        sensor, predictor = get_world_model_by_checkpoint(model_checkpoint)
        filename = 'model_' + args.model_checkpoint + '_{}.txt'
    elif len(args.predictor_checkpoint) > 0:
        predictor_checkpoint = torch.load(os.path.join(PredictorDir, args.predictor_checkpoint + '.pt'), map_location='cpu')
        config = predictor_checkpoint['config']
        predictor = get_predictor_by_checkpoint(predictor_checkpoint)
        sensor_checkpoint = torch.load(os.path.join(SensorDir, config['sensor_checkpoint'] + '.pt'), map_location='cpu')
        sensor = get_sensor_by_checkpoint(sensor_checkpoint)
        filename = 'predictor_' + args.predictor_checkpoint + '_{}.txt'
    else:
        raise ValueError('at least one checkpoint should be given')

    predictor.requires_grad_(False)
    sensor.requires_grad_(False)

    # config dataset
    if len(args.serial_agent_checkpoint) == 0:
        config['preload'] = False
        config['batch_length'] = 100
        _, data_loader, _, _ = load_predictor_dataset(config)
    data_iter = step_loader(data_loader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sensor = sensor.to(device)
    predictor = predictor.to(device)

    emb_losses = []
    obs_losses = []
    reward_losses = []
    for i in tqdm(range(args.batchs)):
        batch = next(data_iter)
        obs = batch['image'].permute(1, 0, 2, 3, 4).to(device).contiguous()
        action = batch['action'].permute(1, 0, 2).to(device).contiguous()
        reward = batch['reward'].permute(1, 0).unsqueeze(dim=-1).to(device).contiguous()

        T, B = obs.shape[:2]
        obs = obs.view(T * B, *obs.shape[2:])
        emb = sensor.encode(obs, output_dist=True).mode().view(T, B, -1)

        if config['predictor'] == 'RAR':
            states = []

            # compute state1 by assuming emb -1 is the same as emb 0, and take no action 
            state = predictor.rnn_cell(torch.cat([emb[0], torch.zeros_like(action[0])], dim=1)) 

            for i in range(T):
                states.append(state)

                _action = action[i]

                emb_dist = predictor.emb_pre(state)

                if i < 10:
                    _emb = emb[i]
                else:
                    _emb = emb_dist.mode()

                state = predictor.rnn_cell(torch.cat([_emb, _action], dim=1), state) # compute next state

        elif config['predictor'] == 'RSAR':
            h, s = predictor._reset(emb[0])

            states = []
            posterior_dists = []
            prior_dists = []

            for i in range(T):
                if i < 10:
                    h, s, _ = predictor.obs_step(h, s, action[i], emb[max(i-1, 0)])
                else:
                    h, s, _ = predictor.img_step(h, s, action[i])

                state = torch.cat([h, s], dim=1)
                states.append(state)
                
        elif config['predictor'] == 'RSSM':
            h, s = predictor._reset(emb[0])
            state = torch.cat([h, s], dim=1)

            states = []

            for i in range(T):
                _action = action[i]
                if i < T:
                    _emb = emb[i]
                    h, s, _, _ = predictor.obs_step(h, s, _action, _emb)
                else:
                    h, s, _ = predictor.img_step(h, s, _action)

                state = torch.cat([h, s], dim=1)
                states.append(state)

        prediction = {"state" : torch.stack(states)}

        states = torch.cat(states, dim=0).contiguous()
        emb_dist = predictor.emb_pre(states)
        prediction['emb'] = emb_dist.mode().view(T, B, *emb.shape[2:])
        
        reward_dist = predictor.reward_pre(states)
        prediction['reward'] = reward_dist.mode().view(T, B, 1)

        pre_emb = prediction['emb']
        pre_emb = pre_emb.view(T * B, pre_emb.shape[-1])
        pre_obs = sensor.decode(pre_emb)
        pre_obs = pre_obs.view(T, B, *pre_obs.shape[1:])
        obs = obs.view(T, B, *obs.shape[1:])

        obs_loss = (obs - pre_obs) ** 2 / 2 + LOG2PI
        obs_losses.append(torch.sum(obs_loss, dim=(2, 3, 4)).cpu())

        emb_loss = (emb - prediction['emb']) ** 2 / 2 + LOG2PI
        emb_losses.append(torch.sum(emb_loss, dim=-1).cpu())
        reward_loss = (reward - prediction['reward']) ** 2 / 2 + LOG2PI
        reward_losses.append(torch.squeeze(reward_loss, dim=-1).cpu())
    
    reward_losses = torch.cat(reward_losses, dim=1).numpy().T
    np.savetxt(os.path.join(output_folder, filename.format('reward')), reward_losses)

    emb_losses = torch.cat(emb_losses, dim=1).numpy().T
    np.savetxt(os.path.join(output_folder, filename.format('emb')), emb_losses)

    obs_losses = torch.cat(obs_losses, dim=1).numpy().T
    np.savetxt(os.path.join(output_folder, filename.format('obs')), obs_losses)

    fig, ax = plt.subplots()
    tsplot(ax, reward_losses)

    fig, ax = plt.subplots()
    tsplot(ax, emb_losses)

    fig, ax = plt.subplots()
    tsplot(ax, obs_losses)

    plt.show()
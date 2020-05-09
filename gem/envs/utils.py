import os, time
import torch

from .wrapper import *

from ..utils import save_npz

def make_env(config):
    suite, task = config['env'].split('_', 1)
    if suite == 'dmc':
        from .dmc import DeepMindControl
        env = DeepMindControl(task)
        env = ActionRepeat(env, config['action_repeat'])
        env = NormalizeActions(env)
    elif suite == 'atari':
        from .artai import Atari
        env = Atari(
            task, config['action_repeat'], (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        env = OneHotAction(env)
    elif suite == 'robos':
        from .robos import Robosuite
        env = Robosuite(task)
        env = ActionRepeat(env, config['action_repeat'])
    else:
        raise NotImplementedError(suite)

    env = TimeLimit(env, config['max_length'] / config['action_repeat'])

    callbacks = []
    if 'datadir' in config.keys():
        datadir = config['datadir']
        callbacks.append(lambda ep: save_episodes(datadir, ep))
        env = Collect(env, callbacks)
        env = RewardObs(env)
    return env

def make_imagine_env_from_predictor(predictor_checkpoint_file, with_emb=True):
    from .imagine import Imagine
    from gem.models.sensor.run_utils import get_sensor_by_checkpoint
    from gem.models.sensor.config import SensorDir
    from gem.models.predictor.run_utils import get_predictor_by_checkpoint
    from gem.models.predictor.config import PredictorDir
    predictor_checkpoint = torch.load(os.path.join(PredictorDir, predictor_checkpoint_file + '.pt'), map_location='cpu')
    predictor = get_predictor_by_checkpoint(predictor_checkpoint)
    predictor.requires_grad_(False)
    sensor_checkpoint = torch.load(os.path.join(SensorDir, predictor_checkpoint['config']['sensor_checkpoint'] + '.pt'), map_location='cpu')
    sensor = get_sensor_by_checkpoint(sensor_checkpoint)
    sensor.requires_grad_(False)
    return Imagine(sensor, predictor, with_emb=with_emb)

def make_imagine_env_from_model(model_checkpoint_file, with_emb=True):
    from .imagine import Imagine
    from gem.models.mix.run_utils import get_world_model_by_checkpoint
    from gem.models.mix.config import ModelDir
    model_checkpoint = torch.load(os.path.join(ModelDir, model_checkpoint_file + '.pt'), map_location='cpu')
    sensor, predictor = get_world_model_by_checkpoint(model_checkpoint)
    sensor.requires_grad_(False)
    predictor.requires_grad_(False)
    return Imagine(sensor, predictor, with_emb=with_emb)

def get_buffer(config):
    from gem.data.buffer import Buffer
    from .wrapper import Collect
    env = make_env(config)
    buffer = Buffer(config['buffer_size'])

    env = Collect(env, [buffer.add])

    action_space = env.action_space

    print('Prefilling Buffer ......')
    for i in range(config['prefill']):
        env.reset()

        num = 0
        start_time = time.time()
        
        while True:
            obs, action, done, info = env.step(action_space.sample())
            num += 1

            if done:
                print(f'use {time.time() - start_time} s to generate sequence {i} with length {num}')
                break
    
    return buffer


def save_episodes(datadir, ep):
    traj_id = len(os.listdir(datadir))
    length = list(ep.values())[0].shape[0]
    save_npz(os.path.join(datadir, f'traj_{traj_id}_{length}.npz'), ep)

def random_generate(env, traj_num):
    action_space = env.action_space

    for i in range(traj_num):
        env.reset()

        num = 0
        start_time = time.time()
        
        while True:
            obs, action, done, info = env.step(action_space.sample())
            num += 1

            if done:
                print(f'use {time.time() - start_time} s to generate sequence {i} with length {num}')
                break
import os

from .wrapper import *

from .artai import Atari
from .dmc import DeepMindControl

from ..utils import save_npz

def make_env(config):
    suite, task = config['env'].split('_', 1)
    if suite == 'dmc':
        env = DeepMindControl(task)
        env = ActionRepeat(env, config['action_repeat'])
        env = NormalizeActions(env)
    elif suite == 'atari':
        env = Atari(
            task, config['action_repeat'], (64, 64), grayscale=False,
            life_done=True, sticky_actions=True)
        env = OneHotAction(env)
    else:
        raise NotImplementedError(suite)

    env = TimeLimit(env, config['max_length'] / config['action_repeat'])

    callbacks = []
    if 'datadir' in config.keys():
        datadir = config['datadir']
        callbacks.append(lambda ep: save_episodes(datadir, [ep]))
        env = Collect(env, callbacks)
        env = RewardObs(env)
    return env

def save_episodes(datadir, ep):
    traj_id = len(os.listdir(datadir))
    save_npz(os.path.join(datadir, f'traj_{traj_id}.npz'), ep)

def random_generate(env, traj_num):
    action_space = env.action_space

    for _ in range(traj_num):
        env.reset()

        num = 0
        
        while True:
            obs, action, done, info = env.step(action_space.sample())
            num += 1

            if done:
                print(f'generate a sequence with length {num}')
                break
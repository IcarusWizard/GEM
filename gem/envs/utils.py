import os, time

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
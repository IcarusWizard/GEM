import os, re
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.serial.trainer import SerialAgentTrainer
from gem.serial.run_utils import config_serial_agent, get_serial_agent_by_checkpoint
from gem.serial.config import get_default_serial_agent_config, SerialLogDir, SerialDir
from gem.utils import setup_seed, create_dir, parse_args, load_npz
from gem.envs.utils import save_episodes, make_env, get_buffer
from gem.envs.imagine import Imagine
from gem.envs.wrapper import Collect

if __name__ == '__main__':
    args = parse_args(get_default_serial_agent_config())

    config = args.__dict__

    if len(config['serial_agent_checkpoint']) > 0: # resume training

        # load checkpoint
        checkpoint = torch.load(os.path.join(SerialDir, config['serial_agent_checkpoint'] + '.pt'), map_location='cpu')
        sensor, predictor, controller = get_serial_agent_by_checkpoint(checkpoint, eval_mode=False)

        last_config = checkpoint['config']
        last_config['start_step'] = last_config['steps']
        last_config['steps'] += config['steps']
        last_config['serial_agent_checkpoint'] = config['serial_agent_checkpoint']
        last_config['gpu'] = config['gpu']

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        last_config['seed'] = seed
        print('using seed {}'.format(seed))

        config = last_config

        world_model = Imagine(sensor, predictor, with_emb=config['with_emb'])

        # get buffer
        buffer = get_buffer(config)
        datafolder = os.path.join(config['log_name'], 'trajs')
        trajs = os.listdir(datafolder)
        trajs.sort(key=lambda x: int(re.findall(r'(\d+)', x)[0]), reverse=True)
        trajs = trajs[:config['buffer_size']]
        for traj in reversed(trajs):
            data = load_npz(os.path.join(datafolder, traj))
            buffer.add(data)

        # config environment
        test_env = make_env(config)
        collect_env = make_env(config)
        callbacks = [
            lambda ep: save_episodes(datafolder, ep),
            buffer.add,
        ]
        collect_env = Collect(collect_env, callbacks)

        # training
        trainer = SerialAgentTrainer(world_model, controller, test_env, collect_env, buffer, config)
        trainer.train()

        # save final checkpoint
        os.system('mv {} {}'.format(
            os.path.join(SerialDir, '{}.pt'.format(config['serial_agent_checkpoint'])),
            os.path.join(SerialDir, '{}.pt.old'.format(config['serial_agent_checkpoint']))
        ))
        trainer.save(os.path.join(SerialDir, '{}.pt'.format(config['serial_agent_checkpoint'])))

    else: # train from scratch
        
        config['start_step'] = 0

        create_dir(SerialDir)

        # setup random seed
        seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
        setup_seed(seed) 
        config['seed'] = seed
        print('using seed {}'.format(seed))

        # get buffer
        buffer = get_buffer(config)

        sample_data = buffer.sample(1, 1)
        config['predict_reward'] = 'reward' in sample_data.keys()
        config['action_dim'] = sample_data['action'].shape[-1]

        # config models
        world_model, controller, sensor_param, predictor_param, controller_param, filename = config_serial_agent(config)
        
        # config environment
        config['sensor_param'] = sensor_param
        config['predictor_param'] = predictor_param
        config['controller_param'] = controller_param
        config['log_name'] = os.path.join(SerialLogDir, filename)
        test_env = make_env(config)
        collect_env = make_env(config)
        datafolder = os.path.join(config['log_name'], 'trajs')
        create_dir(datafolder)
        callbacks = [
            lambda ep: save_episodes(datafolder, ep),
            buffer.add,
        ]
        collect_env = Collect(collect_env, callbacks)

        # training
        trainer = SerialAgentTrainer(world_model, controller, test_env, collect_env, buffer, config)
        trainer.train()
        trainer.save(os.path.join(SerialDir, '{}.pt'.format(filename)))
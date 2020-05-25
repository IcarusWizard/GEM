import os
import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision
from tqdm import tqdm

from gem.controllers.run_utils import config_controller
from gem.controllers.config import get_default_controller_config, ControllerDir, ControllerLogDir
from gem.data import load_predictor_dataset, load_serial_agent_dataset
from gem.envs.utils import make_imagine_env_from_predictor, make_imagine_env_from_model, make_imagine_env_from_serial_agent, make_env, get_buffer
from gem.envs.wrapper import Collect
from gem.utils import setup_seed, create_dir, parse_args

if __name__ == '__main__':
    args = parse_args(get_default_controller_config())

    config = args.__dict__

    create_dir(ControllerDir)

    # setup random seed
    seed = args.seed if not args.seed == -1 else np.random.randint(0, 100000)
    setup_seed(seed) 
    config['seed'] = seed
    print('using seed {}'.format(seed))
    
    # get environment
    if len(config['serial_agent_checkpoint']) > 0:
        world_model = make_imagine_env_from_serial_agent(config['serial_agent_checkpoint'], with_emb=config['with_emb'])
    elif len(config['model_checkpoint']) > 0:
        world_model = make_imagine_env_from_model(config['model_checkpoint'], with_emb=config['with_emb'])
    else:
        world_model = make_imagine_env_from_predictor(config['predictor_checkpoint'], with_emb=config['with_emb'])
    test_env = make_env(config)
    config['state_dim'] = world_model.state_dim
    config['action_dim'] = world_model.action_dim

    
    if config['use_buffer']:
        # get buffer
        buffer = get_buffer(config)
        observation_loader = None
        collect_env = make_env(config)
        collect_env = Collect(collect_env, [buffer.add])
    else:
        # config dataset
        if len(config['serial_agent_checkpoint']) > 0:
            observation_loader = load_serial_agent_dataset(config['serial_agent_checkpoint'], config)
        else:
            _, observation_loader, _, _ = load_predictor_dataset(config)
        buffer = None
        collect_env = None

    controller, controller_param, filename = config_controller(config)

    config['controller_param'] = controller_param
    config['log_name'] = os.path.join(ControllerLogDir, filename)

    trainer_class = controller.get_trainer()

    trainer = trainer_class(controller, world_model, test_env, collect_env, buffer, observation_loader, config)
    trainer.train()
    trainer.save(os.path.join(ControllerDir, '{}.pt'.format(filename)))
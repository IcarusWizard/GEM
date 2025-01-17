import importlib
import torch

def config_controller(config):
    model_param = {
        'state_dim' : config['state_dim'],
        "action_dim" : config['action_dim'],
        "features" : config['controller_features'],
        "hidden_layers" : config['controller_hidden_layers'],
        "actor_mode" : config['actor_mode'],
    }

    module = importlib.import_module('gem.controllers')
    model_class = getattr(module, config['controller'])
    model = model_class(**model_param)

    if len(config['serial_agent_checkpoint']) > 0:
        checkpoint_name = config['serial_agent_checkpoint']
    elif len(config['model_checkpoint']) > 0:
        checkpoint_name = config['model_checkpoint']
    else:
        checkpoint_name = config['predictor_checkpoint']

    filename = config['controller'] + '_' + checkpoint_name
    if len(config['suffix']) > 0:
        filename = filename + '_' + config['suffix']

    return model, model_param, filename

def get_controller_by_checkpoint(checkpoint, eval_mode=True):
    state_dict = checkpoint['controller_state_dict']
    model_param = checkpoint['controller_parameters']
    name = checkpoint['config']['controller']

    module = importlib.import_module('gem.controllers')
    model_class = getattr(module, name)
    model = model_class(**model_param)

    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()

    return model
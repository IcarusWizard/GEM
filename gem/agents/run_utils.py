import importlib
import torch

def config_agent(config):
    model_param = {
        'state_dim' : config['state_dim'],
        "action_dim" : config['action_dim'],
        "features" : config['features'],
        "hidden_layers" : config['hidden_layers'],
        "actor_mode" : config['actor_mode'],
    }

    module = importlib.import_module('gem.agents')
    model_class = getattr(module, config['model'])
    model = model_class(**model_param)

    filename = config['model'] + '_' + config['checkpoint']
    if len(config['suffix']) > 0:
        filename = filename + '_' + len(config['suffix'])

    return model, model_param, filename

def get_agent_by_checkpoint(checkpoint):
    state_dict = checkpoint['model_state_dict']
    model_param = checkpoint['model_parameters']
    name = checkpoint['config']['model']

    module = importlib.import_module('gem.models.agent')
    model_class = getattr(module, name)
    model = model_class(**model_param)

    model.load_state_dict(state_dict)
    model.eval()

    return model
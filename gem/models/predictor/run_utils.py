import importlib
import torch

def config_predictor(config):
    model_param = {
        'obs_dim' : config['latent_dim'],
        "action_dim" : config['action_dim'],
        "hidden_dim" : config['hidden_dim'],
        "action_mimic" : config['action_mimic'],
        "predict_reward" : config['predict_reward'],
        "decoder_config" : {
            "hidden_layers" : config['decoder_hidden_layers'],
            "features" : config['decoder_features'],
            "activation" : config['decoder_activation']
        }
    }

    if config['model'] == 'RSSM':
        model_param['stoch_dim'] = config['stoch_dim']

    module = importlib.import_module('gem.models.predictor')
    model_class = getattr(module, config['model'])
    model = model_class(**model_param)

    filename = config['model'] + '_' + config['checkpoint']
    if len(config['suffix']) > 0:
        filename = filename + '_' + len(config['suffix'])

    return model, model_param, filename

def get_predictor_by_checkpoint(checkpoint):
    state_dict = checkpoint['model_state_dict']
    model_param = checkpoint['model_parameters']
    name = checkpoint['config']['model']

    module = importlib.import_module('gem.models.predictor')
    model_class = getattr(module, name)
    model = model_class(**model_param)

    model.load_state_dict(state_dict)
    model.eval()

    return model
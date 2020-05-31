import importlib
import torch

def config_predictor(config):
    predictor_param = {
        'emb_dim' : config['latent_dim'],
        "action_dim" : config['action_dim'],
        "hidden_dim" : config['state_hidden_dim'],
        "action_mimic" : config['action_mimic'],
        "actor_mode" : config['actor_mode'],
        "predict_reward" : config['predict_reward'],
        "decoder_config" : {
            "hidden_layers" : config['decoder_hidden_layers'],
            "features" : config['decoder_features'],
            "activation" : config['decoder_activation']
        }
    }

    if config['predictor'] == 'RSSM':
        predictor_param['stoch_dim'] = config['state_stoch_dim']
        predictor_param['free_nats'] = config['free_nats']
        predictor_param['kl_scale'] = config['kl_scale']
    elif config['predictor'] == 'RAR':
        predictor_param['warm_up'] = config['warm_up']

    module = importlib.import_module('gem.models.predictor')
    model_class = getattr(module, config['predictor'])
    model = model_class(**predictor_param)

    filename = config['predictor'] + '_' + config['sensor_checkpoint']
    if len(config['suffix']) > 0:
        filename = filename + '_' + config['suffix']

    return model, predictor_param, filename

def get_predictor_by_checkpoint(checkpoint, eval_mode=True):
    state_dict = checkpoint['predictor_state_dict']
    model_param = checkpoint['predictor_parameters']
    name = checkpoint['config']['predictor']

    module = importlib.import_module('gem.models.predictor')
    model_class = getattr(module, name)
    model = model_class(**model_param)

    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()

    return model
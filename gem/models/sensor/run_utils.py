from . import *

import torch

def config_model_train(config, model_param):
    model_param['network_type'] = config['network_type']

    if config['network_type']== 'mlp':
        model_param['config'] = {
            "features" : config['features'], 
            "hidden_layers" : config['hidden_layers'],
        }
    else:
        if config['network_type'] == 'conv':
            model_param['config'] = {
                "conv_features" : config['conv_features'],
                "down_sampling" : config['down_sampling'],
                "batchnorm" : config['use_batchnorm'],
                "mlp_features" : config['features'],
                "mlp_layers" : config['hidden_layers'],
            } 
        elif config['network_type'] == 'fullconv':
            model_param['config'] = {
                "conv_features" : config['conv_features'],
                "down_sampling" : config['down_sampling'],
                "batchnorm" : config['use_batchnorm'],
            } 

        assert len(config['res_layers']) == 1 or len(config['res_layers']) == config['down_sampling']
        if len(config['res_layers']) == 1:
            model_param['config']['res_layers'] = config['res_layers'] * config['down_sampling']
        else:
            model_param['config']['res_layers'] = config['res_layers']

    if config['model']== 'VAE' or config['model'] == 'CVAE' or config['model'] == 'AVAE':
        model_param.update({
            "latent_dim" : config['latent_dim'],
            "output_type" : config['output_type'],
            "use_mce" : config['use_mce'],
            "output_type" : config['output_type'],
        })       

        if config['model']== 'VAE':
            model = VAE(**model_param)
        elif config['model'] == 'CVAE':
            model = CVAE(**model_param)
        elif config['model'] == 'AVAE':
            model = AVAE(**model_param)

    elif config['model']== 'FVAE':
        model_param.update({
            "latent_dim" : config['latent_dim'],
            "flow_features" : config['flow_features'],
            "flow_hidden_layers" : config['flow_hidden_layers'],
            "flow_num_transformation" : config['flow_num_transformation'],
        })        
        model = FVAE(**model_param)
    elif config['model']== 'VQ-VAE':
        if not config['network_type']== 'fullconv':
            model_param['config']['latent_dim'] = config['latent_dim']
        model_param.update({
            "k" : config['k'],
            "d" : config['d'],
            "beta" : config['beta'],
            "output_type" : config['output_type'],
        })     
        model = VQ_VAE(**model_param)
    else:
        raise ValueError('Model {} is not supported!'.format(config['model']))

    return model, model_param

def get_model_by_checkpoint(checkpoint):
    state_dict = checkpoint['model_state_dict']
    model_param = checkpoint['model_parameters']
    name = checkpoint['config']['model']

    if name == 'VAE':
        model = VAE(**model_param)
    elif name == 'FVAE':
        model = FVAE(**model_param)
    elif name == 'VQ-VAE':
        model = VQ_VAE(**model_param)
    elif name == 'CVAE':
        model = CVAE(**model_param)
    elif name == 'AVAE':
        model == AVAE(**model_param)

    model.load_state_dict(state_dict)
    model.eval()

    return model

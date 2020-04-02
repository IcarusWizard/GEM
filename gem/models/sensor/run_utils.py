import importlib

def config_sensor(config, model_param):
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
    elif config['model']== 'FVAE' or config['model'] == 'PFVAE':
        model_param.update({
            "latent_dim" : config['latent_dim'],
            "flow_features" : config['flow_features'],
            "flow_hidden_layers" : config['flow_hidden_layers'],
            "flow_num_transformation" : config['flow_num_transformation'],
        })        
    elif config['model']== 'VQ-VAE':
        if not config['network_type']== 'fullconv':
            model_param['config']['latent_dim'] = config['latent_dim']
        model_param.update({
            "k" : config['k'],
            "d" : config['d'],
            "beta" : config['beta'],
            "output_type" : config['output_type'],
        })     
    else:
        raise ValueError('Model {} is not supported!'.format(config['model']))

    module = importlib.import_module('gem.models.sensor')
    model_class = getattr(module, config['model'])
    model = model_class(**model_param)

    return model, model_param

def get_sensor_by_checkpoint(checkpoint):
    state_dict = checkpoint['model_state_dict']
    model_param = checkpoint['model_parameters']
    name = checkpoint['config']['model']

    module = importlib.import_module('gem.models.sensor')
    model_class = getattr(module, name)
    model = model_class(**model_param)

    model.load_state_dict(state_dict)
    model.eval()

    return model
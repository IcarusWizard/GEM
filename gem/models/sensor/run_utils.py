import importlib

def config_sensor(config, sensor_param):
    sensor_param['network_type'] = config['network_type']
    sensor_param['kl_scale'] = config['kl_scale']

    if config['network_type'] == 'conv':
        sensor_param['config'] = {
            "conv_features" : config['conv_features'],
            "activation" : config['conv_activation'],
            "down_sampling" : config['down_sampling'],
            "batchnorm" : config['use_batchnorm'],
            "mlp_features" : config['sensor_features'],
            "mlp_layers" : config['sensor_hidden_layers'],
        } 
    elif config['network_type'] == 'fullconv':
        sensor_param['config'] = {
            "conv_features" : config['conv_features'],
            "down_sampling" : config['down_sampling'],
            "batchnorm" : config['use_batchnorm'],
        } 

    assert len(config['res_layers']) == 1 or len(config['res_layers']) == config['down_sampling']
    if len(config['res_layers']) == 1:
        sensor_param['config']['res_layers'] = config['res_layers'] * config['down_sampling']
    else:
        sensor_param['config']['res_layers'] = config['res_layers']

    if config['sensor']== 'VAE' or config['sensor'] == 'CVAE' or config['sensor'] == 'AVAE' or config['sensor'] == 'SWVAE':
        sensor_param.update({
            "latent_dim" : config['latent_dim'],
            "output_type" : config['output_type']
        })       
    elif config['sensor']== 'FVAE' or config['sensor'] == 'PFVAE':
        sensor_param.update({
            "latent_dim" : config['latent_dim'],
            "output_type" : config['output_type'],
            "flow_config" : {
                "features" : config['flow_features'],
                "hidden_layers" : config['flow_hidden_layers'],
                "num_transformation" : config['flow_num_transformation'],
            }
        })        
    else:
        raise ValueError('Sensor model {} is not supported!'.format(config['sensor']))

    module = importlib.import_module('gem.models.sensor')
    model_class = getattr(module, config['sensor'])
    model = model_class(**sensor_param)

    filename = "{}_{}".format(config['sensor'], config['dataset'])

    if len(config['suffix']) > 0:
        filename = filename + '_' + config['suffix']

    return model, sensor_param, filename

def get_sensor_by_checkpoint(checkpoint, eval_mode=True):
    state_dict = checkpoint['sensor_state_dict']
    sensor_param = checkpoint['sensor_parameters']
    name = checkpoint['config']['sensor']

    module = importlib.import_module('gem.models.sensor')
    model_class = getattr(module, name)
    model = model_class(**sensor_param)

    model.load_state_dict(state_dict)
    if eval_mode:
        model.eval()

    return model
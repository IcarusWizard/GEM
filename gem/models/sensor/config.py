import os
from gem.utils import AttrDict

SensorLogDir = os.path.join('logs', 'sensor')
SensorDir = os.path.join('checkpoint', 'sensor')

def get_default_sensor_config():
    config = AttrDict()
    
    # base config for sensor model
    config.sensor_checkpoint = ''
    config.dataset = 'dmc_finger_spin'
    config.sensor = 'VAE'
    config.latent_dim = 64
    config.free_nats = 3.0
    config.kl_scale = 1.0
    config.output_type = 'fix_std'
    config.network_type = 'conv'

    # network config for MLP
    config.sensor_hidden_layers = 0
    config.sensor_features = 256

    # network config for Conv
    config.conv_features = 256
    config.down_sampling = 4
    config.res_layers = [0]
    config.use_batchnorm = False
    config.conv_activation = 'relu'

    # flow config
    config.flow_features = 256
    config.flow_hidden_layers = 3
    config.flow_num_transformation = 8

    # adversarial training config
    config.n_critic = 1

    # training config
    config.seed = -1 
    config.batch_size = 64
    config.image_per_file = 8
    config.preload = True
    config.gpu = '0'
    config.workers = 9999
    config.steps = 100000
    config.m_lr = 1e-3
    config.m_beta1 = 0.9
    config.m_beta2 = 0.999
    config.m_grad_clip = 1000.0

    # log config
    config.log_step = 1000
    config.suffix = ''

    return config
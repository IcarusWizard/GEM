import os
from gem.utils import AttrDict

PredictorLogDir = os.path.join('logs', 'predictor')
PredictorDir = os.path.join('checkpoint', 'predictor')

def get_default_predictor_config():
    config = AttrDict()
    
    # base config
    config.predictor = 'RSSM'
    config.sensor_checkpoint = 'VAE_dmc_finger_spin'
    config.state_hidden_dim = 256
    config.state_stoch_dim = 32
    config.decoder_hidden_layers = 2
    config.decoder_features = 256
    config.decoder_activation = 'elu'
    config.action_mimic = True
    config.actor_mode = 'continuous'

    # training config
    config.seed = -1 
    config.batch_size = 50
    config.batch_length = 50
    config.fix_start = False
    config.preload = True
    config.gpu = '0'
    config.workers = 9999
    config.steps = 50000
    config.m_lr = 4e-6
    config.m_beta1 = 0.9
    config.m_beta2 = 0.999
    config.grad_clip = 100.0

    # log config
    config.log_step = 1000
    config.fps = 20
    config.suffix = ''

    return config
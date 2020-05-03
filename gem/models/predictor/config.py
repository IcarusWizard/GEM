import os
from gem.utils import AttrDict

PredictorLogDir = os.path.join('logs', 'predictor')
PredictortDir = os.path.join('checkpoint', 'predictor')

def get_default_predictor_config():
    config = AttrDict()
    
    # base config
    config.model = 'RSSM'
    config.checkpoint = 'VAE_dmc_finger_spin'
    config.hidden_dim = 256
    config.stoch_dim = 32
    config.decoder_hidden_layers = 2
    config.decoder_features = 256
    config.decoder_activation = 'elu'
    config.action_mimic = False

    # training config
    config.seed = -1 
    config.batch_size = 50
    config.horizon = 50
    config.fix_start = False
    config.preload = True
    config.gpu = '0'
    config.workers = 9999
    config.steps = 50000
    config.lr = 1e-3
    config.beta1 = 0.9
    config.beta2 = 0.999

    # log config
    config.log_step = 1000
    config.fps = 15
    config.suffix = ''

    return config
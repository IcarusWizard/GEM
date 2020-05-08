import os
from gem.utils import AttrDict

ControllerLogDir = os.path.join('logs', 'controller')
ControllerDir = os.path.join('checkpoint', 'controller')

def get_default_controller_config():
    config = AttrDict()

    # base controller config
    config.predictor_checkpoint = 'RSSM_VAE_dmc_finger_spin'
    config.dataset = 'dmc_finger_spin'
    config.controller = 'VGC'
    config.controller_hidden_layers = 3
    config.controller_features = 256
    config.actor_mode = 'continuous'
    config.gamma = 0.99
    config['lambda'] = 0.95
    config.horizon = 15

    # env config
    config.env = 'dmc_finger_spin'
    config.action_repeat = 2
    config.max_length = 1000

    # training config
    config.seed = -1 
    config.batch_size = 128
    config.image_per_file = 2
    config.preload = True
    config.gpu = '0'
    config.workers = 9999
    config.steps = 100000
    config.c_lr = 8e-5
    config.c_beta1 = 0.9
    config.c_beta2 = 0.999
    config.grad_clip = 100.0

    # log config
    config.log_step = 1000
    config.fps = 20
    config.suffix = ''

    return config
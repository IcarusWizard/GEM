from gem.utils import AttrDict

def get_default_agent_config():
    config = AttrDict()

    config.checkpoint = 'RSSM_VAE_dmc_finger_spin'
    config.dataset = 'dmc_finger_spin'
    config.model = 'ACAgent'
    config.hidden_layers = 3
    config.features = 256
    config.actor_mode = 'continuous'

    config.gamma = 0.99
    config['lambda'] = 0.95
    config.horizon = 15

    config.env = 'dmc_finger_spin'
    config.action_repeat = 2
    config.max_length = 1000

    config.seed = -1 
    config.batch_size = 128
    config.image_per_file = 2
    config.preload = True
    config.gpu = '0'
    config.workers = 9999
    config.steps = 100000
    config.lr = 1e-3
    config.beta1 = 0.9
    config.beta2 = 0.999

    config.log_step = 1000
    config.fps = 15
    config.suffix = ''

    return config
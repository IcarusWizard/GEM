import os
from gem.utils import AttrDict

SerialLogDir = os.path.join('logs', 'serial_agent')
SerialDir = os.path.join('checkpoint', 'serial_agent')

def get_default_serial_agent_config():
    config = AttrDict()

    from gem.models.mix.config import get_default_world_model_config
    from gem.controllers.config import get_default_controller_config

    config.update(get_default_controller_config())
    config.update(get_default_world_model_config())

    # overwrite training config
    config.batch_size = 50
    config.batch_length = 50
    config.log_step = 100

    return config
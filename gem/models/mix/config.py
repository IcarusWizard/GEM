import os
from gem.utils import AttrDict

ModelLogDir = os.path.join('logs', 'model')
ModelDir = os.path.join('checkpoint', 'model')

def get_default_world_model_config():
    config = AttrDict()

    from gem.models.sensor.config import get_default_sensor_config
    from gem.models.predictor.config import get_default_predictor_config

    config.update(get_default_sensor_config())
    config.update(get_default_predictor_config())

    return config
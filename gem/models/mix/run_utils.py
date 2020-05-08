import importlib
import torch

from gem.models.sensor.run_utils import config_sensor, get_sensor_by_checkpoint
from gem.models.predictor.run_utils import config_predictor, get_predictor_by_checkpoint

def config_world_model(config, dataset_config):
    sensor_param = {
        "c" : dataset_config['obs'][0],
        "h" : dataset_config['obs'][1],
        "w" : dataset_config['obs'][2],
    }

    sensor, sensor_param, _ = config_sensor(config, sensor_param)
    predictor, predictor_param, _ = config_predictor(config)

    filename = config['predictor'] + '_' + config['sensor'] + '_' + config['dataset']
    if len(config['suffix']) > 0:
        filename = filename + '_' + config['suffix']

    return sensor, predictor, sensor_param, predictor_param, filename

def get_world_model_by_checkpoint(checkpoint):
    sensor = get_sensor_by_checkpoint(checkpoint)
    predictor = get_predictor_by_checkpoint(checkpoint)

    return sensor, predictor
import importlib
import torch

from gem.models.sensor.run_utils import config_sensor, get_sensor_by_checkpoint
from gem.models.predictor.run_utils import config_predictor, get_predictor_by_checkpoint
from gem.controllers.run_utils import config_controller, get_controller_by_checkpoint
from gem.envs.imagine import Imagine

def config_serial_agent(config):
    sensor_param = {
        "c" : config['image_channel'],
        "h" : config['image_size'],
        "w" : config['image_size'],
    }

    sensor, sensor_param, _ = config_sensor(config, sensor_param)
    predictor, predictor_param, _ = config_predictor(config)
    world_model = Imagine(sensor, predictor, with_emb=config['with_emb'])
    config['state_dim'] = world_model.state_dim
    config['action_dim'] = world_model.action_dim
    controller, controller_param, _ = config_controller(config)

    filename = '_'.join([config['controller'], config['predictor'], config['sensor'], config['env']])
    if len(config['suffix']) > 0:
        filename = filename + '_' + config['suffix']

    return world_model, controller, sensor_param, predictor_param, controller_param, filename

def get_serial_agent_by_checkpoint(checkpoint, eval_mode=True):
    sensor = get_sensor_by_checkpoint(checkpoint, eval_mode=eval_mode)
    predictor = get_predictor_by_checkpoint(checkpoint, eval_mode=eval_mode)
    controller = get_controller_by_checkpoint(checkpoint, eval_mode=eval_mode)

    return sensor, predictor, controller
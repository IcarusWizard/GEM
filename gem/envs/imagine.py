import numpy as np
import gym

class Imagine:

    def __init__(self, sensor, predictor):
        super().__init__()
        self.sensor = sensor
        self.predictor = predictor
        self.state_dim = predictor.state_dim
        self.action_dim = predictor.action_dim

        self.dtype = next(self.sensor.parameters()).dtype
        self.device = next(self.sensor.parameters()).device

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, [self.state_dim], dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, [self.action_dim], dtype=np.float32)

    def step(self, action, obs=None):
        if obs is not None:
            emb = self.sensor.encode(obs)
        else:
            emb = None
        self.state, reward = self.predictor.step(self.state, action, emb)

        info = None
        done = False

        return self.state, reward, done, info

    def reset(self, obs):
        emb = self.sensor.encode(obs)
        self.state = self.predictor.reset(emb)
        return self.state

    def render(self, state):
        with_time = len(state.shape) == 3
        if with_time:
            T, B, S = state.shape
            state = state.view(-1, S)
        emb = self.predictor.obs_pre(state).mode()
        obs = self.sensor.decode(emb)
        if with_time:
            obs = obs.view(T, B, *obs.shape[1:])
        return obs

    def to(self, device):
        self.sensor = self.sensor.to(device)
        self.predictor = self.predictor.to(device)
        self.device = device

        return self
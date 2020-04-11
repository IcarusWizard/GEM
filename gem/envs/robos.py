import numpy as np
import robosuite as suite
import gym

ROBOS_TASKS = [
    'SawyerLift',
    'SawyerStack',
    'BaxterPegInHole',
    'BaxterLift',
    'SawyerPickPlace',
    'SawyerNutAssembly',
]

class Robosuite:

    def __init__(self, task, size=(128, 128), camera='frontview'):
        self._size = size
        self._camera = camera
        assert task in ROBOS_TASKS, f"There is no task names f{task}, please select from one of {ROBOS_TASKS}"
        self._env = suite.make(
            task,
            has_renderer=False,          # no on-screen renderer
            has_offscreen_renderer=True, # off-screen renderer is required for camera observations
            ignore_done=True,            # (optional) never terminates episode
            use_camera_obs=True,         # use camera observations
            camera_height=size[0],            # set camera height
            camera_width=size[1],             # set camera width
            camera_name=camera,         # select camera
            use_object_obs=False,        # no object feature when training on pixels
            reward_shaping=True          # (optional) using a shaping reward
        )    

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        low, high = self._env.action_spec
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if self._camera == 'frontview':
            obs['image'] = obs['image'][::-1] # flip the image (bug in robosuite)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        if self._camera == 'frontview':
            obs['image'] = obs['image'][::-1] # flip the image (bug in robosuite)
        return obs

    def render(self, *args, **kwargs):
        self._env.render()
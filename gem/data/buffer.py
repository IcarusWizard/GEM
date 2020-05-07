import torch
import numpy as np
import random

class Buffer:
    def __init__(self):
        self.trajs = []

    def _process_data(self, data):
        if len(data.shape) == 4:
            data = data / 255.0
            data = np.transpose(data, (0, 3, 1, 2))
        return torch.as_tensor(data, dtype=torch.float32).contiguous()

    def add(self, traj):
        for k, v in traj.items():
            traj[k] = self._process_data(v)
        self.trajs.append(traj)

    def _split(self, traj, horizon):
        length = list(traj.values())[0].shape[0]
        start = random.randint(0, length - horizon)
        end = start + horizon

        return {k : v[start:end] for k, v in traj.items()}

    def sample(self, batch_size, horizon):
        trajs = random.choices(self.trajs, k=batch_size)
        trajs = [self._split(traj, horizon) for traj in trajs]
        trajs = {k : torch.stack([traj[k] for traj in trajs]) for k in trajs[0].keys()}
        return trajs
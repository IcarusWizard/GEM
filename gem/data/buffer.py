import torch
import numpy as np
import random

class Buffer:
    def __init__(self, buffer_size):
        self.trajs = [None for _ in range(buffer_size)]
        self.index = 0
        self.max_index = 0
        self.buffer_size = buffer_size

    def _process_data(self, data):
        if len(data.shape) == 4:
            data = data / 255.0 - 0.5
            data = np.transpose(data, (0, 3, 1, 2))
        return torch.as_tensor(data, dtype=torch.float32).contiguous()

    def add(self, traj):
        for k, v in traj.items():
            traj[k] = self._process_data(v)
        self.trajs[self.index] = traj
        self.index = (self.index + 1) % self.buffer_size
        self.max_index = min(self.max_index + 1, self.buffer_size)

    def _split(self, traj, batch_length):
        length = list(traj.values())[0].shape[0]
        start = random.randint(0, length - batch_length)
        end = start + batch_length

        return {k : v[start:end] for k, v in traj.items()}

    def sample(self, batch_size, batch_length):
        trajs = random.choices(self.trajs[:self.max_index], k=batch_size)
        trajs = [self._split(traj, batch_length) for traj in trajs]
        trajs = {k : torch.stack([traj[k] for traj in trajs]) for k in trajs[0].keys()}
        return trajs

    def sample_image(self, batch_size):
        trajs = random.choices(self.trajs[:self.max_index], k=batch_size)
        imgs = [traj['image'] for traj in trajs]
        imgs = torch.stack([img[random.randint(0, img.shape[0] - 1)] for img in imgs])
        return (imgs, )

    def generator(self, batch_size, batch_length=None):
        while True:
            if batch_length is not None:
                yield self.sample(batch_size, batch_length)
            else:
                yield self.sample_image(batch_size)
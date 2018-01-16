import math
import random
import torch
from torch.utils import data

class SpiralDataset(data.Dataset):

    def __init__(self, n_points=500, noise=0.2):
        self.points = torch.Tensor(n_points, 7)
        self.labels = torch.IntTensor(n_points)

        n_positive = n_points // 2
        n_negative = n_points = n_positive

        for i, point in enumerate(self._gen_spiral_points(n_positive, 0, noise)):
            self.points[i], self.labels[i] = point, 1

        for i, point in enumerate(self._gen_spiral_points(n_negative, math.pi, noise)):
            self.points[i+n_positive] = point
            self.labels[i+n_positive] = -1


    def _gen_spiral_points(self, n_points, delta_t, noise):
        for i in range(n_points):
            r = i / n_points * 5
            t = 1.75 * i / n_points * 2 * math.pi + delta_t
            x = r * math.sin(t) + random.uniform(-1, 1) * noise
            y = r * math.cos(t) + random.uniform(-1, 1) * noise
            yield torch.Tensor([x, y, x**2, y**2, x*y, math.sin(x), math.sin(y)])


    def __len__(self):
        return len(self.labels)


    def __getitem__(self):
        pass


    def to_numpy(self):
        return self.points.numpy(), self.labels.numpy()

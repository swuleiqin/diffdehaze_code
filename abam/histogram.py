import math

import torch
from torch import nn


class SoftHistogram(nn.Module):
    def __init__(self, bins=64, sigma=0.02, epsilon=1e-6):
        super().__init__()
        if bins < 2 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("histogram requires bins>=2 and finite sigma>0")
        if not math.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("histogram epsilon must be in (0,1)")
        self.bins, self.sigma, self.epsilon = bins, sigma, epsilon
        self.register_buffer("centers", torch.linspace(0, 1, bins))

    def forward(self, illumination):
        values = illumination.float().flatten(1)

        counts = torch.stack([
            torch.exp(-0.5 * ((values - center) / self.sigma).square()).mean(dim=1)
            for center in self.centers
        ], dim=1)
        counts = counts + self.epsilon
        return counts / counts.sum(dim=1, keepdim=True)

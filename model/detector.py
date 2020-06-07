import torch
from torch import nn
import torch.nn.functional as F


class Detector(nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.detect = nn.Sequential(
            nn.Linear(config['detect_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.detect(x)
        return out

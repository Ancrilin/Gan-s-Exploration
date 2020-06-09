import torch
from torch import nn
import torch.nn.functional as F


class Detector(nn.Module):
    def __init__(self, config):
        super(Detector, self).__init__()
        self.model = self.model = nn.Sequential(
            nn.Linear(config['feature_dim'], config['detect_dim']),
            nn.Sigmoid()
        )
        self.detect = nn.Sequential(
            nn.Linear(config['detect_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_vector = self.model(x)
        out = self.detect(f_vector)
        return out


class Detector_v2(nn.Module):
    def __init__(self, config):
        super(Detector_v2, self).__init__()
        self.model = self.model = nn.Sequential(
            nn.Linear(config['feature_dim'], config['detect_dim']),
            nn.Sigmoid()
        )
        self.detect = nn.Sequential(
            nn.Linear(config['detect_dim'], 2),
        )

    def forward(self, x):
        f_vector = self.model(x)
        out = self.detect(f_vector)
        return out

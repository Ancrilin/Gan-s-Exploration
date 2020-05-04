import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, config: dict):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config['feature_dim'], config['D_Wf_dim']),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(config['D_Wf_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_vector = self.model(x)
        discriminator_output = self.discriminator(f_vector)
        return discriminator_output, f_vector


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config['G_z_dim'], 1024, bias=False),
            nn.Tanh(),
            nn.Linear(1024, 768, bias=False),
            nn.Tanh(),
            nn.Linear(768, config['feature_dim'], bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        # [batch, feature_dim]
        feature_vector = self.model(z)
        return feature_vector
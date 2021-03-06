# coding: utf-8
# @author: Ross
# @file: GAN.py 
# @time: 2020/01/13
# @contact: devross@gmail.com
from torch import nn


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

        self.classifier = nn.Sequential(
            nn.Linear(config['D_Wf_dim'], config['n_class']),
        )

    def forward(self, x, return_feature=False):
        # [batch, Wf]
        f_vector = self.model(x)
        # [batch, 1]
        discriminator_output = self.discriminator(f_vector)
        # [batch, n_class]
        classification_output = self.classifier(f_vector)
        if return_feature:
            return f_vector, discriminator_output, classification_output
        return discriminator_output, classification_output

    def detect_only(self, x, return_feature=False):
        """只进行OOD判别"""
        # [batch, Wf]
        f_vector = self.model(x)
        # [batch, 1]
        discriminator_output = self.discriminator(f_vector)
        if return_feature:
            return f_vector, discriminator_output
        return discriminator_output

    def get_vector(self, x):
        return self.model(x)


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


if __name__ == '__main__':
    D_config = {'feature_dim': 768, 'Wf_dim': 512, 'n_class': 2}
    D = Discriminator(D_config)
    print(D)

    G_config = {'feature_dim': 768, 'z_dim': 2}
    G = Generator(G_config)
    print(G)

# coding: utf-8
# @author: Ross
# @file: GAN.py
# @time: 2020/01/13
# @contact: devross@gmail.com
import torch
from torch import nn
import torch.nn.functional as F


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


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)
        self.embed = 768
        self.dropout = 0.5
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, config['G_z_dim'])) for k in self.filter_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), config['feature_dim'])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, z):
        # [batch, feature_dim]
        # feature_vector = self.model(z)
        # battch, seq, feature_dim
        # batch, channel, seq, feature_dim
        # channel 为 1，要增加 1个维度
        z = z.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(z, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    D_config = {'feature_dim': 768, 'Wf_dim': 512, 'n_class': 2}
    D = Discriminator(D_config)
    print(D)

    G_config = {'feature_dim': 768, 'z_dim': 2}
    G = Generator(G_config)
    print(G)

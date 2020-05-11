import torch
import torch.nn as nn
from transformers import BertModel
from configparser import SectionProxy


class Pos(nn.Module):
    def __init__(self, config):
        super(Pos, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(2 * config['pos_dim'], config['feature_dim'], bias=False),
            nn.Tanh(),
        )
        self.model2 = nn.Sequential(
            nn.Linear(2 * config['feature_dim'], 2 * config['feature_dim'], bias=False),
            nn.Tanh(),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(2 * config['feature_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, pos1, pos2, bert_feature):
        out = self.model1(torch.cat([pos1, pos2], -1))
        out = self.model2(torch.cat([out, bert_feature], -1))
        out = self.discriminator(out)
        return out



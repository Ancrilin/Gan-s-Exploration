import torch
import torch.nn as nn
from transformers import BertModel
from configparser import SectionProxy
import numpy as np


class Pos_emb(nn.Module):
    def __init__(self, config):
        super(Pos_emb, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['n_pos'], config['embed'], padding_idx=0)
        # self.w = nn.Parameter(torch.Tensor(config[hidden_size * 2))
        self.model = nn.Sequential(
            nn.Linear(config['pos_dim'] + 1, config['feature_dim'], bias=False),
            nn.Tanh(),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2 * config['feature_dim'], 1),
            nn.Sigmoid()
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed'], nhead=config['nhead'])
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])

    def forward(self, pos1, pos2, bert_feature):
        pos = self.get_embedding(pos1, pos2)
        pos_feature = self.transformer_encoder(pos)
        pos_feature = self.model(pos_feature)
        out = self.discriminator(torch.cat((bert_feature, pos_feature), dim=-1))
        return out

    def get_embedding(self, pos1, pos2):
        embed = self.embedding(pos2)
        embed = torch.add(embed, self.pos_embedding(self.config['embed'], self.config['maxlen']))
        final = torch.rand(self.config['batch_size'], self.config['maxlen'], self.config['embed'])
        for i in range(self.config['batch_size']):
            for index, j in enumerate(pos1[i]):
                if j[1] == 0:
                    break
                for k in range(j[0].numpy(), j[1].numpy()):
                    final[i][k] = embed[i][index]
        cls = self.embedding(torch.tensor([1]))
        cls = cls.repeat(self.config['batch_size'], 1, 1)
        final = torch.cat((cls, final), dim=1)
        return final.to(self.config['device'])

    def pos_embedding(self, embed, pad_size):
        pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        return pe.to(self.config['device'])



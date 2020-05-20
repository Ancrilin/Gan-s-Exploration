import torch
import torch.nn as nn
from transformers import BertModel
from configparser import SectionProxy
import numpy as np
import os
import traceback


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Pos_emb(nn.Module):
    def __init__(self, config):
        super(Pos_emb, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config['n_pos'], config['pos_dim'], padding_idx=0)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['pos_dim'], nhead=config['nhead'])
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config['num_layers'])
        self.discriminator = nn.Sequential(
            nn.Linear(config['feature_dim'] + config['pos_dim'], 1),
            nn.Sigmoid()
        )

    def forward(self, pos1, pos2, bert_feature):
        pos = self.get_embedding(pos1, pos2)
        pos_feature = self.transformer_encoder(pos)[:, 0]           # 取综合的特征
        out = self.discriminator(torch.cat((bert_feature, pos_feature), dim=-1))
        return out

    def get_embedding(self, pos1, pos2):
        embed = self.embedding(pos2)
        embedding = embed + self.pos_embedding(self.config['pos_dim'], self.config['maxlen']).to(self.config['device'])
        final = torch.rand(len(pos1), self.config['maxlen'], self.config['pos_dim']).to(self.config['device'])
        for i in range(len(pos1)):
            for index, j in enumerate(pos1[i]):
                if j[1] == 0:
                    for m in range(pos1[i][index - 1][1], self.config['maxlen']):
                        final[i][m] = self.embedding(torch.LongTensor([0]).to(self.config['device']))  # padding
                    break
                for k in range(j[0].data.cpu().numpy(), j[1].data.cpu().numpy()):
                    final[i][k] = embedding[i][index]
        cls = self.embedding(torch.LongTensor([1]).to(self.config['device']))
        cls = cls.repeat(len(pos1), 1, 1)
        final = torch.cat((cls, final), dim=1)
        return final.to(self.config['device'])

    def pos_embedding(self, embed, pad_size):
        pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        return pe



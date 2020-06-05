# coding: utf-8
# @author: Ross
# @file: BERT.py
# @time: 2020/01/13
# @contact: devross@gmail.com
from configparser import SectionProxy

import torch
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """BERT分类器"""

    def __init__(self, config: SectionProxy, d_config: dict):
        super(BertClassifier, self).__init__()
        self.config = config
        self.d_config = d_config
        self.bert = BertModel.from_pretrained(config['PreTrainModelDir'])

        self.model = nn.Sequential(
            nn.Linear(d_config['feature_dim'], d_config['D_Wf_dim']),
            nn.Sigmoid()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(d_config['D_Wf_dim'], 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_config['D_Wf_dim'], d_config['n_class']),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_feature=False):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        real_feature = sequence_output
        f_vector = self.model(real_feature)
        discriminator_output = self.discriminator(f_vector)
        classification_output = self.classifier(f_vector)
        if return_feature:
            return f_vector, discriminator_output, classification_output
        return discriminator_output, classification_output

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    import os
    import sys
    import numpy as np

    sys.path.append(os.getcwd())
    config = {'PreTrainModelDir': 'bert/bert-base-uncased'}
    model = BertClassifier(config, 3)
    ids = np.random.random_integers(0, 20000, size=(10, 32))
    ids = torch.LongTensor(ids)
    print(torch.argmax(model(ids), 1))

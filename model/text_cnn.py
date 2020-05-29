#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by Ross on 2020/3/11
from configparser import SectionProxy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class TextCNN(nn.Module):
    """BERT分类器"""

    def __init__(self, config: SectionProxy, num_labels):
        super(TextCNN, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(config['PreTrainModelDir'])
        embedding_dim = self.bert.config.hidden_size
        chanel_num = 1
        filter_num = 150
        filter_sizes = [3, 4, 5]

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dim)) for size in filter_sizes])
        self.dropout = nn.Dropout(0.8)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        x = sequence_output.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

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
    model = TextCNN(config, 3)
    ids = np.random.random_integers(0, 20000, size=(10, 32))
    ids = torch.LongTensor(ids)
    print(torch.argmax(model(ids), 1))

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by Ross on 2020/3/12
from configparser import SectionProxy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel


class BertBiLSTM(nn.Module):
    """BERT分类器"""

    def __init__(self, config: SectionProxy, num_labels):
        super(BertBiLSTM, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(config['PreTrainModelDir'])

        embedding_dim = self.bert.config.hidden_size
        self.lstm_size = 128

        self.bilstm = nn.LSTM(embedding_dim, self.lstm_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.lstm_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        lengths = torch.sum(attention_mask, 1).long()
        packed = pack_padded_sequence(sequence_output, lengths, batch_first=True, enforce_sorted=False)
        lstm_output, (h_n, c_n) = self.bilstm(packed)
        final_output = torch.cat([h_n[0, :, :], h_n[1, :, :]], 1)
        logits = self.fc(final_output)
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
    model = BertBiLSTM(config, 3)
    ids = np.random.random_integers(0, 20000, size=(10, 32))
    ids = torch.LongTensor(ids)
    # mask = torch.ones(10, 32, dtype=torch.long)
    mask = torch.LongTensor([[1] * 16 + [0] * 16 for i in range(10)])
    type_ids = torch.ones(10, 32, dtype=torch.long)
    print(torch.argmax(model(ids, type_ids, mask), 1))

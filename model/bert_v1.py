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

    def __init__(self, config: SectionProxy, num_labels):
        super(BertClassifier, self).__init__()
        self.config = config
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(config['PreTrainModelDir'])
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(pooled_output)
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
    model = BertClassifier(config, 3)
    ids = np.random.random_integers(0, 20000, size=(10, 32))
    ids = torch.LongTensor(ids)
    print(torch.argmax(model(ids), 1))

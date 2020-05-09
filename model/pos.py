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
        out = self.model1(torch.cat([pos1, pos2], 0))
        out = self.model2(torch.cat([out, bert_feature], 0))
        out = self.discriminator(out)
        return out


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

class Pos_Bert(nn.Module):
    def __init__(self, config, bert_config: SectionProxy, num_labels):
        super(Pos_Bert, self).__init__()
        self.pos = Pos(config)
        self.bert = BertModel.from_pretrained(bert_config['PreTrainModelDir'])
        self.classifier = torch.nn.Linear(bert_config.hidden_size, num_labels)


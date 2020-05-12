# coding: utf-8
# @author: Ross
"""oos-eval數據集预处理"""

from processor.base_processor import BertProcessor

"""将数据集做简单的预处理，包括生成类标签文件"""
import os
import json
from config import Config
from configparser import SectionProxy

import jieba.posseg as pseg
import jieba


class PosSMPProcessor(BertProcessor):
    """oos-eval 数据集处理"""

    def __init__(self, bert_config, maxlen=32):
        super(PosSMPProcessor, self).__init__(bert_config, maxlen)
        self.pos = {"pad": 0}

    def convert_to_ids(self, dataset: list) -> list:
        ids_data = []
        for line in dataset:
            ids_data.append(self.parse_line(line))
        return ids_data

    def read_dataset(self, path: str, data_types: list):
        """
        读取数据集文件
        :param path: 路径
        :param data_type: [type1, type2]
        :return dataset list
        """
        # load dataset
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for data_type in data_types:
            for line in data[data_type]:
                dataset.append(line)
        return dataset

    def load_label(self, path):
        """load label"""
        with open(path, 'r', encoding='utf-8') as f:
            self.id_to_label = json.load(f)
            self.label_to_id = {label: i for i, label in enumerate(self.id_to_label)}

    def load_pos(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            t_pos = json.load(fp)
        for key in t_pos:
            if key not in self.pos:
                self.pos[key] = len(self.pos)

    def parse_line(self, line: list) -> list:
        """
        :param line: [text, label]
        :return: [text_ids, mask, type_ids, label_ids]
        """
        text = line['text']
        label = line['domain']

        ids = self.parse_text_to_bert_token(text) + [self.parse_label(label)] + self.pos_tagging(text, self.maxlen)
        return ids

    def pos_tagging(self, text, maxlen):
        cut_ids = []
        tok = jieba.tokenize(text)
        label = 1
        for tk in tok:
            cut_ids.append([tk[1], tk[2]])
            # for i in range(tk[1], tk[2]):
            #     cut_ids.append(label)
            label = (1 if label == 2 else 2)
        if len(cut_ids) < maxlen:
            cut_ids += [[0, 0]] * (maxlen - len(cut_ids))
        tags = []
        for word, tag in pseg.lcut(text, use_paddle=True):
            if tag not in self.pos:
                self.pos[tag] = len(self.pos)
            tags.append(self.pos[tag])
        if len(tags) < maxlen:
            tags += ([0] * (maxlen - len(tags)))
        else:
            tags = tags[:maxlen]
        pos_mask = [1] * len(text) + [0] * (maxlen - len(text))
        cut_ids = cut_ids[:32]
        tags = tags[:32]
        pos_mask = pos_mask[:32]
        return [cut_ids, tags, pos_mask]

    def parse_text(self, text) -> (list, list, list):
        """
        将文本转为ids
        :param text: 字符串文本
        :return: [token_ids, mask, type_ids]
        """
        return self.parse_text_to_bert_token(text)

    def parse_label(self, label):
        """
        讲label转为ids
        :param label: 文本label
        :return: ids
        """
        return self.label_to_id[label]


# --------------------oos-eval-------------------- #

def preprocess_smp_eval(config: SectionProxy):
    data_dir = config['DataDir']
    files = os.listdir(data_dir)
    for file in files:
        # if not data file, skip
        if not file.endswith('.json'):
            continue

        # if output file exists, skip
        output_file = file.replace('.json', '.label')
        if os.path.exists(os.path.join(data_dir, output_file)):
            continue

        labels = set()
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # for k, v in data.items():
        #     for line in v:
        #         labels.add(line[1])
        for line in data['train']:
            labels.add(line['domain'])

        # 对类按照字典序排序后，将oos放在最后面
        try:
            labels.remove('chat')
        except KeyError:
            pass

        labels = ['chat'] + sorted(labels)

        with open(os.path.join(data_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)


config = Config('config/data.ini')
oos_config = config('smp')
preprocess_smp_eval(oos_config)

# --------------------oos-eval-------------------- #

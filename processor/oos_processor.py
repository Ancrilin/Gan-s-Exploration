# coding: utf-8
# @author: Ross
# @file: oos_processor.py 
# @time: 2020/01/13
# @contact: devross@gmail.com
"""oos-eval數據集预处理"""

from processor.base_processor import BertProcessor
import json

"""将数据集做简单的预处理，包括生成类标签文件"""
import os
import json
from config import Config
from configparser import SectionProxy
import numpy as np


class OOSProcessor(BertProcessor):
    """oos-eval 数据集处理"""

    def __init__(self, bert_config, maxlen=32):
        super(OOSProcessor, self).__init__(bert_config, maxlen)

    def convert_to_ids(self, dataset: list) -> list:
        ids_data = []
        for line in dataset:
            ids_data.append(self.parse_line(line))
        return ids_data

    def read_dataset(self, path: str, data_types: list, mode=0, maxlen=-1, minlen=-1, pre_exclude=False):
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
                if pre_exclude:
                    if maxlen != -1 and len(line[0]) > maxlen:
                        continue
                    if minlen != -1 and len(line[0]) <= minlen:
                        continue
                dataset.append(line)
        return dataset

    def load_label(self, path):
        """load label"""
        with open(path, 'r', encoding='utf-8') as f:
            self.id_to_label = json.load(f)
            self.label_to_id = {label: i for i, label in enumerate(self.id_to_label)}

    def parse_line(self, line: list) -> list:
        """
        :param line: [text, label]
        :return: [text_ids, mask, type_ids, label_ids]
        """
        text = line[0]
        label = line[-1]

        ids = self.parse_text_to_bert_token(text) + [self.parse_label(label)]
        return ids

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

def preprocess_oos_eval(config: SectionProxy):
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
            labels.add(line[1])

        # 对类按照字典序排序后，将oos放在最后面
        try:
            labels.remove('oos')
        except KeyError:
            pass

        labels = ['oos'] + sorted(labels)

        with open(os.path.join(data_dir, output_file), 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)


config = Config('config/data.ini')
oos_config = config('oos-eval')
preprocess_oos_eval(oos_config)

# --------------------oos-eval-------------------- #

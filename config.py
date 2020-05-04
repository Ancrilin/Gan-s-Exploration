# coding: utf-8
# @author: Ross
# @file: config.py 
# @time: 2020/01/12
# @contact: devross@gmail.com
import json
import os
from configparser import ConfigParser, SectionProxy


class Config:
    """配置处理基类"""

    config_parser = ConfigParser()

    def __init__(self, path):
        self.parse(path)

    def parse(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.config_parser.read_file(f)

    def __call__(self, *args, **kwargs) -> SectionProxy:
        """pass the section name and return config for a section"""
        return self.config_parser[args[0]]


class BertConfig(Config):

    def __init__(self, path):
        super().__init__(path)

    def __call__(self, *args, **kwargs) -> SectionProxy:
        # 要先转成dict，python3.6的bug，3.7修复
        param = super().__call__(*args, **kwargs)

        file = os.path.join(param['PreTrainModelDir'], 'config.json')
        with open(file, 'r', encoding='utf-8') as f:
            for k, v in json.load(f).items():
                param.__setattr__(k, v)
        return param


if __name__ == '__main__':
    # bert_config = Config('config/bert.ini')
    # print(bert_config('bert-base-chinese'))
    # bert_config = BertConfig('config/bert.ini')
    # bert_config = bert_config('bert-base-uncased')
    config = Config('config/bert.ini')
    bert_config = config('bert-base-uncased')
    print(bert_config['PreTrainModelDir'])
    print(config.config_parser.sections())
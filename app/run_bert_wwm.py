from transformers import BertModel, BertTokenizer


if __name__ == '__main__':
    MODEL_NAME = 'hfl/chinese-bert-wwm'
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
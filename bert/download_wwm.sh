#!/usr/bin/env bash
save_path="bert/chinese-bert-wwm"
mkdir ${save_path}

echo "downloading BERT to ${save_path}..."

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-bert-wwm/pytorch_model.bin
mv pytorch_model.bin ${save_path}/pytorch_model.bin

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-bert-wwm/config.json
mv config.json ${save_path}/config.json

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/hfl/chinese-bert-wwm/vocab.txt
mv vocab.txt ${save_path}/vocab.txt

exit 0

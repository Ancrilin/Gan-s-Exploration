#!/usr/bin/env bash

if [[ ! $1 ]]; then
    echo "Parse pre-trained BERT type. i.g. [bert-base-uncased | bert-large-uncased | bert-base-chinese]"
    exit 1
fi

model_name=$1
save_path="bert/${model_name}" # 保存模型的文件夹

if [[ ! -d ${save_path} ]]; then
  mkdir ${save_path}
fi

echo "downloading BERT to ${save_path}..."

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/${model_name}-pytorch_model.bin
mv  ${model_name}-pytorch_model.bin ${save_path}/pytorch_model.bin

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/${model_name}-config.json
mv  ${model_name}-config.json ${save_path}/config.json

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/${model_name}-vocab.txt
mv  ${model_name}-vocab.txt ${save_path}/vocab.txt

echo "done"
exit 0

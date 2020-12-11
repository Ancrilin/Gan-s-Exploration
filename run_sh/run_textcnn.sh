#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
# dataset_file="binary_smp_full_v2"
for seed in ${seeds} ; do
  python -m app.run_textcnn \
    --n_epoch=15 \
    --patience=5 \
    --train_batch_size=32 \
    --predict_batch_size=32 \
    --bert_type=bert-base-chinese \
    --fine_tune \
    --lr=2e-6 \
    --dataset=smp \
    --data_file=$1 \
    --output_dir=output/text-cnn_s$seed \
    --gradient_accumulation_steps=1 \
    --do_train \
    --do_eval \
    --do_test \
    --seed=$seed \
    --result=$2
  rm -rf output/text-cnn_s$seed/save
done
exit 0

#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
# dataset_file="binary_smp_full_v2"
for seed in ${seeds} ; do
  python -m app.run_bert \
    --n_epoch=30 \
    --patience=5 \
    --bert_type=bert-base-chinese \
    --fine_tune \
    --dataset=smp \
    --data_file=$1 \
    --output_dir=output/bert_s$seed \
    --do_train \
    --do_eval \
    --do_test \
    --seed=$seed \
  
  rm -rf output/bert_$seed/save
done
exit 0

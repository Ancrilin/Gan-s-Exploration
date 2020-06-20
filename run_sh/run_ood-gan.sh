#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
# dataset_file="binary_smp_full_v2"
for seed in ${seeds} ; do
  !python -m app.run_gan \
    --D_lr=2e-5 \
    --G_lr=2e-5 \
    --bert_lr=2e-5 \
    --fine_tune \
    --n_epoch=500 \
    --patience=10 \
    --train_batch_size=32 \
    --bert_type=bert-base-chinese \
    --dataset=smp \
    --data_file=$1 \
    --output_dir=output/ood-gan_s$seed \
    --do_train \
    --do_eval \
    --do_test \
    --do_vis \
    --feature_dim=1024 \
    --G_z_dim=1024  \
    --seed=$seed  \
    --result=$2
  rm -rf output/ood-gan_s$seed/save
done
exit 0

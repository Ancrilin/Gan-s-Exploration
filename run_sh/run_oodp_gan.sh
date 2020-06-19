#! /bin/bash

seeds="16 123"
# dataset_file="binary_smp_full_v2"
for seed in ${seeds} ; do
  python -m app.run_oodp_gan \
  --model=gan  \
  --seed=${seed}  \
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
  --output_dir=oodp-gan/oodp-gan-smp_maxlen-1_mode1_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --mode=1  \
  --maxlen=$2 \
  --result=$3
done
exit 0

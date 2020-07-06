#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
## dataset_file="binary_true_smp_full_v2"
## $1 dataset_file
## $2 mode
## $3 maxlen
## $4 minlen
## $5 optim_mode
## $6 length_weight
## $7 sample_weight
## $8 gross_result name
for seed in ${seeds} ; do
  python -m app.run_oodp_gan_length \
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
  --output_dir=oodp-gan/oodp-gan-smp_maxlen$3_minlen$4_mode$2_optim_mode$5_lw$6_sw$7_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=768 \
  --G_z_dim=1024  \
  --mode=$2  \
  --maxlen=$3 \
  --minlen=$4 \
  --optim_mode=$5 \
  --length_weight=$6 \
  --sample_weight=$7 \
  --result=$8
  rm -rf oodp-gan/oodp-gan-smp_maxlen$3_minlen$4_mode$2_optim_mode$5_lw$6_sw$7_s${seed}/save
done
exit 0

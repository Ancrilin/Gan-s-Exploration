#! /bin/bash

seeds="16  256  1024  2048  8192"
## dataset_file="binary_undersample"
## $1 dataset_file
## $2 maxlen
## $3 minlen
## $4 optim_mode
## $5 length_weight
## $6 gross_result name
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
  --bert_type=bert-large-uncased \
  --dataset=oos-eval \
  --data_file=$1 \
  --output_dir=oodp-gan/oodp-gan-oos_maxlen$2_minlen$3_optim_mode$4_lw$5_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=1024 \
  --G_z_dim=1024  \
  --maxlen=$2 \
  --minlen=$3 \
  --optim_mode=$4 \
  --length_weight=$5 \
  --result=$6
  rm -rf oodp-gan/oodp-gan-oos_maxlen$2_minlen$3_optim_mode$4_lw$5_s${seed}/save
done
exit 0

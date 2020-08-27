#! /bin/bash

seeds="16 123 256 512 1024 1536 2048 4096 8192"
## dataset_file="binary_undersample","binary_wiki_aug"
## $1 dataset_file
## $2 mode
## $3 maxlen
## $4 minlen
## $5 gross_result name
for seed in ${seeds} ; do
  python -m app.run_oodp_gan_oos \
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
  --output_dir=oodp-gan/oos-oodp-gan_mode$2_maxlen$3_minlen$4_s${seed} \
  --do_train \
  --do_eval \
  --do_test \
  --do_vis \
  --feature_dim=1024 \
  --G_z_dim=1024  \
  --mode=$2  \
  --maxlen=$3 \
  --minlen=$4 \
  --result=$5
  rm -rf oodp-gan/oos-oodp-gan_mode$2_maxlen$3_minlen$4_s${seed}/save
done
exit 0

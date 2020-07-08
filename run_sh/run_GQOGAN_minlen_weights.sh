#! /bin/bash

weights="0.2 0.4 0.6 0.8"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 minlen
# $5 optim_mode
# $6 length_weight
# $7 sample_weight
# $8 save_path of my computer
for weight in ${weights} ; do
  bash run_sh/run_oodp_gan_length.sh ${1} ${2} ${3} ${4} ${5} ${weight} ${7} GQOGAN-maxlen${3}_minlen${4}_mode${2}_optim_mode${5}_ls${weight}_sw${7}
  mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${4}_optim_mode${5}_ls${weight}_sw${7}
  mv GQOGAN-maxlen${3}_minlen${4}_mode${2}_optim_mode${5}_ls${weight}_sw${7}_gross_result.csv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${4}_optim_mode${5}_ls${weight}_sw${7}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${4}_optim_mode${5}_ls${weight}_sw${7}" "$8"
done
exit 0

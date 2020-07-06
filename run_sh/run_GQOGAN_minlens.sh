#! /bin/bash

lengths="6 5 4 3 2"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 minlen
# $5 optim_mode
# $6 length_weight
# $7 sample_weight
# $8 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_length.sh ${1} ${2} ${3} ${len} ${5} ${6} ${7} GQOGAN-maxlen${3}_minlen${len}_mode${2}_optim_mode${5}_ls${6}_sw${7}
  mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${len}_optim_mode${5}_ls${6}_sw${7}
  mv GQOGAN-maxlen${3}_minlen${len}_mode${2}_optim_mode${5}_ls${6}_sw${7}_gross_result.csv GQOGAN_mode${2}_maxlen${3}_minlen${len}_optim_mode${5}_ls${6}_sw${7}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${len}_optim_mode${5}_ls${6}_sw${7}" "$8"
done
exit 0

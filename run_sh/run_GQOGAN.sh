#! /bin/bash

lengths="15"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 minlen
# $4 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan.sh ${1} ${2} ${len} $3 GQOGAN-maxlen${len}_minlen-1_mode$2
  mv oodp-gan GQOGAN_mode${2}_maxlen${len}
  mv GQOGAN-maxlen${len}_minlen-1_mode${2}_gross_result.csv GQOGAN_mode${2}_maxlen${len}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${len}" "$4"
done
exit 0

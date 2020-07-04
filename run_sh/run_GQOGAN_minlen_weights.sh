#! /bin/bash

weights="0.2 0.3 0.4 0.5 0.6"
# dataset_file="binary_true_smp_full_v2_length"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 minlen
# $5 length_mode
# $6 length_weight
# $7 save_path of my computer
for weight in ${weights} ; do
  bash run_sh/run_oodp_gan_length.sh ${1} ${2} ${3} ${4} ${5} ${weight} GQOGAN-maxlen${3}_minlen${4}_mode${2}_length_mode${5}_weight${weight}
  mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${4}_length_mode${5}_weight${weight}
  mv GQOGAN-maxlen${3}_minlen${4}_mode${2}_length_mode${5}_weight${weight}_gross_result.csv GQOGAN_mode${2}_maxlen${3}_minlen${4}_length_mode${5}_weight${weight}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${4}_length_mode${5}_weight${weight}" "$7"
done
exit 0

#! /bin/bash

lengths="6 5 4 3 2"
# dataset_file="binary_true_smp_full_v2_length"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 minlen
# $5 length_mode
# $6 length_weight
# $7 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_length.sh ${1} ${2} ${3} ${len} ${5} ${6} GQOGAN-maxlen${3}_minlen${len}_mode${2}_length_mode${5}_weight${6}
  mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${len}_length_mode${5}_weight${6}
  mv GQOGAN-maxlen${3}_minlen${len}_mode${2}_length_mode${5}_weight${6}_gross_result.csv GQOGAN_mode${2}_maxlen${3}_minlen${len}_length_mode${5}_weight${6}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${len}_length_mode${5}_weight${6}" "$7"
done
exit 0

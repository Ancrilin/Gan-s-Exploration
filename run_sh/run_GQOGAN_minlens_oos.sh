#! /bin/bash

lengths="-1"
# dataset_file="binary_undersample"
# $1 dataset_file
# $2 maxlen
# $3 minlen
# $4 optim_mode
# $5 length_weight
# $6 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_length_oos.sh ${1} ${2} ${len} ${4} ${5} oos-GQOGAN-maxlen${2}_minlen${len}_optim_mode${4}_ls${5}
  mv oodp-gan oos-GQOGAN-maxlen${2}_minlen${len}_optim_mode${4}_ls${5}
  mv oos-GQOGAN-maxlen${2}_minlen${len}_optim_mode${4}_ls${5}_gross_result.csv oos-GQOGAN-maxlen${2}_minlen${len}_optim_mode${4}_ls${5}
  cp -r "/content/Gan-s-Exploration/oos-GQOGAN-maxlen${2}_minlen${len}_optim_mode${4}_ls${5}" "$6"
done
exit 0

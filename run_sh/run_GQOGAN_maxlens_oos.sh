#! /bin/bash

lengths="16 15 14 13 12 11"
# dataset_file="binary_undersample"
# $1 dataset_file
# $2 maxlen
# $3 minlen
# $4 optim_mode
# $5 length_weight
# $6 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_length_oos.sh ${1} ${len} ${3} ${4} ${5} oos-GQOGAN-maxlen${len}_minlen${3}_optim_mode${4}_ls${5}
  mv oodp-gan oos-GQOGAN-maxlen${len}_minlen${3}_optim_mode${4}_ls${5}
  mv oos-GQOGAN-maxlen${len}_minlen${3}_optim_mode${4}_ls${5}_gross_result.csv oos-GQOGAN-maxlen${len}_minlen${3}_optim_mode${4}_ls${5}
  cp -r "/content/Gan-s-Exploration/oos-GQOGAN-maxlen${len}_minlen${3}_optim_mode${4}_ls${5}" "$6"
done
exit 0

#! /bin/bash

lengths="6"
# dataset_file="binary_undersample","binary_wiki_aug"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_oos.sh ${1} ${2} ${3} ${len} GQOGAN-maxlen${3}_minlen${len}_mode${2}
  mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${len}
  mv GQOGAN-maxlen${3}_minlen${len}_mode${2}_gross_result.csv GQOGAN_mode${2}_maxlen${3}_minlen${len}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${len}" "$4"
done
exit 0

#! /bin/bash

lengths="19 20 21 22 23 24 25"
# dataset_file="binary_undersample","binary_wiki_aug"
# $1 dataset_file
# $2 mode
# $3 minlen
# $4 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan_oos.sh ${1} ${2} ${len} ${3} GQOGAN-maxlen${len}_minlen${3}_mode${2}
  mv oodp-gan GQOGAN_mode${2}_maxlen${len}_minlen${3}
  mv GQOGAN-maxlen${len}_minlen${3}_mode${2}_gross_result.csv GQOGAN_mode${2}_maxlen${len}_minlen${3}
  cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${len}_minlen${3}" "$4"
done
exit 0

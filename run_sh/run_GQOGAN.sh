#! /bin/bash

lengths="6"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 maxlen
# $4 minlen
# $5 save_path of my computer
bash run_sh/run_oodp_gan.sh ${1} ${2} ${3} ${4} GQOGAN-maxlen${3}_minlen${4}_mode${2}
mv oodp-gan GQOGAN_mode${2}_maxlen${3}_minlen${4}
mv GQOGAN-maxlen${3}_minlen${4}_mode${2}_gross_result.csv GQOGAN_mode${2}_maxlen${3}_minlen${4}
cp -r "/content/Gan-s-Exploration/GQOGAN_mode${2}_maxlen${3}_minlen${4}" "$5"
exit 0

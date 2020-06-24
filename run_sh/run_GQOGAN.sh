#! /bin/bash

lengths="21"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 save_path
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan.sh $1 $2 ${len} GQOGAN-maxlen${len}_mode$2
done
mv oodp-gan GQOGAN_mode$2
mv GQOGAN_mode$2 $3/GQOGAN_mode$2
exit 0

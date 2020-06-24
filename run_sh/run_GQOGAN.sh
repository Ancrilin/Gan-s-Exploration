#! /bin/bash

lengths="15 16 17 18 19 20 21 22 23 24 25"
# dataset_file="binary_true_smp_full_v2"
# $1 dataset_file
# $2 mode
# $3 minlen
# $4 save_path of my computer
for len in ${lengths} ; do
  bash run_sh/run_oodp_gan.sh $1 $2 ${len} $3 GQOGAN-maxlen${len}_minlen-1_mode$2
done
mv oodp-gan GQOGAN_mode$2_maxlen${len}
mv GQOGAN-maxlen${len}_minlen-1_mode$2_gross_result.csv GQOGAN_mode$2_maxlen${len}
mv "GQOGAN_mode$2_maxlen${len}" "$4"
exit 0

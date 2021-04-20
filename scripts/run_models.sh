#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
echo set visible gpu as gpu $1

agent=$2
dataset=$3
echo $agent on $dataset

for i in 1 2 3;
do
  save_name=${agent}_${dataset}_${i}
  echo doing $i, saving to $save_name
  python -m src.rl_compiler_opt -e llvm-autophase-ic-v0 -d $dataset -a $agent -t 1000000 -ef 5000 -vd 10 -ed 1000 -s 164 -sn $save_name #> script_outputs/${save_name}.txt 2&>1
done

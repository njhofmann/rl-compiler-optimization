#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

gpu=$1
dataset=$2
seed=$3
trial=$4
iters=$5
save_name=opt_dqn_seed_${seed}_trial_${trial}_dataset_${dataset}_iters_${iters}
echo running $save_name on gpu $gpu

python -m src.rl_compiler_opt -e llvm-autophase-ic-v0 -d $dataset -a dqn -t 1000000 -ef 10000 -vd 10 -ed $iters -s $seed -sn $save_name --batch_size 16 --train_freq 4 --gradient_steps 8 --buffer_size 7000 --target_update_interval 10 -lr 0.0008341182143924176 --max_grad_norm 2.3596893735792785 -g 0.9020378649352845 > script_outputs/${save_name}.txt
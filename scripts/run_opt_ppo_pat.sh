#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

gpu=$1
dataset=$2
seed=$3
trial=$4
pat=$5
save_name=opt_ppo_seed_${seed}_trial_${trial}_dataset_${dataset}_pat_${pat}
echo running $save_name on gpu $gpu

python -m src.rl_compiler_opt -e llvm-autophase-ic-v0 -d $dataset -a ppo -t 1000000 -ef 10000 -vd 10 -ep $pat -s $seed -sn $save_name --batch_size 8 --n_steps 4 --gae_lambda 0.95 --ent_coef 0.0 --vf_coef .5 -lr 0.0001 --max_grad_norm 0.5 -g 0.99 > script_outputs/${save_name}.txt
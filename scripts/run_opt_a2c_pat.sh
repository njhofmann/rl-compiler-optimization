#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

gpu=$1
dataset=$2
seed=$3
trial=$4
pat=$5
save_name=opt_a2c_seed_${seed}_trial_${trial}_dataset_${dataset}_pat_${pat}
echo running $save_name on gpu $gpu

python -m src.rl_compiler_opt -e llvm-autophase-ic-v0 -d $dataset -a a2c -t 1000000 -ef 10000 -vd 10 -ep pat -s $seed -sn $save_name --n_steps 4 --gae_lambda 0.9891141725428422 --ent_coef 0.01191178960492222 --vf_coef 0.36856790773990467 -lr 0.0009537804954927808 --max_grad_norm 9.271137134064423 -g 0.9958692302177554 > script_outputs/${save_name}.txt
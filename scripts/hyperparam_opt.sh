#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

agent=$2
echo hyperparameter opt for $agent on gpu $1

python -m src.hyperparam_opt -t 2 -rs 1 > script_outputs/${agent}_hyperparam_opt.txt #2&>1
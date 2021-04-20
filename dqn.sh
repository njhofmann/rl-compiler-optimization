#!/bin/bash
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 1000000 --eval_freq 5000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_1 > dqn_1.txt 2&>1
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 1000000 --eval_freq 5000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_2 > dqn_1.txt 2&>1
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 1000000 --eval_freq 5000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_3 ? dqn_3.txt 2&>1
#!/bin/bash
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 2000000 --eval_freq 1000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_1
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 2000000 --eval_freq 1000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_2
python -m src.rl_compiler_opt --env llvm-autophase-ic-v0 --datasets blas-v0 --agent dqn -t 2000000 --eval_freq 1000 --eval_dur 10 --eps_dur 1000 --seed 164 -sn dqn_3
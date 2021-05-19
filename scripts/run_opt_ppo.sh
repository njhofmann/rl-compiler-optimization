#!/bin/bash
# gpu dataset seed trial iters/pat

# 100 iters, blas-v0
./scripts/run_opt_ppo_iters.sh 3 blas-v0 2046 0 100
./scripts/run_opt_ppo_iters.sh 3 blas-v0 2046 1 100

./scripts/run_opt_ppo_iters.sh 3 blas-v0 207 0 100
./scripts/run_opt_ppo_iters.sh 3 blas-v0 207 1 100

./scripts/run_opt_ppo_iters.sh 3 blas-v0 4656 0 100
./scripts/run_opt_ppo_iters.sh 3 blas-v0 4656 1 100

# 1000 iters, blas-v0
./scripts/run_opt_ppo_iters.sh 3 blas-v0 2046 0 1000
./scripts/run_opt_ppo_iters.sh 3 blas-v0 2046 1 1000

./scripts/run_opt_ppo_iters.sh 3 blas-v0 207 0 1000
./scripts/run_opt_ppo_iters.sh 3 blas-v0 207 1 1000

./scripts/run_opt_ppo_iters.sh 3 blas-v0 4656 0 1000
./scripts/run_opt_ppo_iters.sh 3 blas-v0 4656 1 1000

# 100 iters, github-v0
./scripts/run_opt_ppo_iters.sh 3 github-v0 2046 0 100
./scripts/run_opt_ppo_iters.sh 3 github-v0 2046 1 100

./scripts/run_opt_ppo_iters.sh 3 github-v0 207 0 100
./scripts/run_opt_ppo_iters.sh 3 github-v0 207 1 100

./scripts/run_opt_ppo_iters.sh 3 github-v0 4656 0 100
./scripts/run_opt_ppo_iters.sh 3 github-v0 4656 1 100

# 1000 iters, github-v0
./scripts/run_opt_ppo_iters.sh 3 github-v0 2046 0 1000
./scripts/run_opt_ppo_iters.sh 3 github-v0 2046 1 1000

./scripts/run_opt_ppo_iters.sh 3 github-v0 207 0 1000
./scripts/run_opt_ppo_iters.sh 3 github-v0 207 1 1000

./scripts/run_opt_ppo_iters.sh 3 github-v0 4656 0 1000
./scripts/run_opt_ppo_iters.sh 3 github-v0 4656 1 1000

# 100 pat, blas-v0
./scripts/run_opt_ppo_pat.sh 3 blas-v0 2046 0 100
./scripts/run_opt_ppo_pat.sh 3 blas-v0 2046 1 100

./scripts/run_opt_ppo_pat.sh 3 blas-v0 207 0 100
./scripts/run_opt_ppo_pat.sh 3 blas-v0 207 1 100

./scripts/run_opt_ppo_pat.sh 3 blas-v0 4656 0 100
./scripts/run_opt_ppo_pat.sh 3 blas-v0 4656 1 100

# 100 pat, github-v0
./scripts/run_opt_ppo_pat.sh 3 github-v0 2046 0 100
./scripts/run_opt_ppo_pat.sh 3 github-v0 2046 1 100

./scripts/run_opt_ppo_pat.sh 3 github-v0 207 0 100
./scripts/run_opt_ppo_pat.sh 3 github-v0 207 1 100

./scripts/run_opt_ppo_pat.sh 3 github-v0 4656 0 100
./scripts/run_opt_ppo_pat.sh 3 github-v0 4656 1 100

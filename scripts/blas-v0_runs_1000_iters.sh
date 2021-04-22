#!/bin/bash
./scripts/run_models_iters.sh 0 dqn blas-v0 1000 &
./scripts/run_models_iters.sh 1 a2c blas-v0 1000 &
./scripts/run_models_iters.sh 2 ppo blas-v0 1000 &

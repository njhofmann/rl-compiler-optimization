#!/bin/bash
./scripts/run_models.sh 0 dqn blas-v0 &
./scripts/run_models.sh 1 a2c blas-v0 &
./scripts/run_models.sh 2 ppo blas-v0 &

#!/bin/bash
./scripts/hyperparam_opt.sh 0 dqn &
./scripts/hyperparam_opt.sh 1 a2c &
./scripts/hyperparam_opt.sh 2 ppo &

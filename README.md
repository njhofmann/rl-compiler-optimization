# rl-compiler-optimization

Study of the application of deep reinforcement learning to the phase ordering problem for compiler optimization, using 
newly released CompilerGym toolkit and deep RL agents from StableBaselines v3.

## Motivation

The phase ordering problem is finding the optimal sequence of optimizations to apply to some program being compiled to 
maximize or minimize some attribute of the program such as execution time, energy consumption, executable size, etc.

Often the order in which such optimization operations are applied is fixed or based off of heuristics, so the sequence 
applied is likely not optimal leaving room on the table for further improvement via "smarter" orderings.

Previous works have looked methods such as neuroevolution or deep RL (in limited contexts) to find such better orderings.
This project aims to see if a reinforcement learning agent can be trained on this task.

## Background

We use implementations of deep reinforcement learning methods from StableBaselines3 (PPO, DQN, and A2C), in environments 
provided by the CompilerGym toolkit that reframe compiler optimization as a OpenAI Gym environment. We utilize Bayesian
hyperparameter optimization from the Ray Tune library to find a generally good set of hyperparams for each architecture.

## Results

## Usage

Run main program with `python -m src.rl_compiler_opt [args]`, list main args with `-h`

## Dependencies

Require the following dependencies:

- CompilerGym 0.1.8
- StableBaselines3 1.3.0
- RayTune 2.0.0
- Matplotlib 3.4.1

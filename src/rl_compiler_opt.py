import argparse as ap
import random as r
from typing import Union, List, Tuple, Optional
import numpy as np

import gym as g
import stable_baselines3 as sb
import compiler_gym.datasets as cg

from src import paths as p, arg_parser as apr, gym_wrapper as gy

Agent = Union[sb.DQN, sb.A2C, sb.PPO]


def init_agent(agent_type: str, env: g.Env) -> Agent:
    if agent_type == 'dqn':
        # policy = sb.dqn.MlpPolicy(observation_space=env.observation_space, action_space=env.action_space)
        return sb.DQN(policy='MlpPolicy', env=env, verbose=1, buffer_size=2000, learning_starts=1000,
                      learning_rate=5e-4, gradient_steps=4, target_update_interval=10, batch_size=32,
                      exploration_fraction=.1, max_grad_norm=40, exploration_final_eps=.02,
                      policy_kwargs={'net_arch': [256]})
    elif agent_type == 'a2c':
        return sb.A2C(policy='MlpPolicy', env=env, verbose=1, max_grad_norm=40, learning_rate=.0001, vf_coef=.5,
                      ent_coef=.01, gae_lambda=1.0)
    elif agent_type == 'ppo':
        return sb.PPO(policy='MlpPolicy', env=env, verbose=1, batch_size=10, policy_kwargs={'net_arch': [64, 64]})
    raise ValueError(f'agent {agent_type} is not a supported agent')


def save_agent(agent: Agent, name: str) -> None:
    p.AGENTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent.save(p.AGENTS_DIRC / f'{name}.zip')


def move_eval_results(model_name: str) -> None:
    old_file = p.RESULTS_DIRC / 'evaluations.npz'
    if not old_file.exists():
        raise RuntimeError(f'old evaluations file wasn\'t saved')

    new_file = p.RESULTS_DIRC / f'{model_name}_eval_results.npz'
    old_file.rename(new_file)
    return new_file


def get_and_set_benchmarks(train_env: gy.CompilerGymWrapper, eval_env: gy.CompilerGymWrapper, datasets: List[str],
                           eval_datasets: Optional[List[str]], overlap: bool) -> Tuple[gy.CompilerGymWrapper, gy.CompilerGymWrapper]:
    def get_benchmarks(dataset: str) -> List[str]:
        return [benchmark for benchmark in train_env.compiler_env.benchmarks if dataset in benchmark]

    if eval_datasets is None:
        eval_datasets = datasets

    train_env.compiler_env.require_datasets(datasets)
    eval_env.compiler_env.require_datasets(datasets)

    if eval_datasets:
        eval_env.compiler_env.require_datasets(eval_datasets)

    datasets, eval_datasets = set(datasets), set(eval_datasets)
    train_benchmarks, test_benchmarks = set(), set()

    if overlap:
        overlap_datasets = datasets.union(eval_datasets)
        datasets.difference_update(overlap_datasets)
        eval_datasets.difference_update(overlap_datasets)

        for dataset in overlap_datasets:
            benchs = get_benchmarks(dataset)
            r.shuffle(benchs)
            mid = round(len(benchs) / 2)
            front, back = benchs[:mid], benchs[mid:]
            train_benchmarks.update(front)
            test_benchmarks.update(back)

    for dataset in datasets:
        train_benchmarks.update(get_benchmarks(dataset))

    for dataset in eval_datasets:
        test_benchmarks.update(get_benchmarks(dataset))

    train_env.set_benchmarks(list(train_benchmarks))
    eval_env.set_benchmarks(list(test_benchmarks))
    return train_env, eval_env


def train_and_eval_agent(args: ap.Namespace) -> None:
    save_name = args.save_name
    train_log_path = None
    if args.log_training:
        train_log_path = p.TRAIN_LOGS / f'{save_name}_training.txt'
    env = gy.CompilerGymWrapper(compiler_env=args.env,
                                eps_iters=args.eps_dur,
                                eps_runtime=args.eps_runtime,
                                eps_patience=args.eps_patience,
                                random_seed=args.seed,
                                logging_path=train_log_path,
                                k_prev_actions=args.k_prev_actions)

    eval_env = gy.CompilerGymWrapper(compiler_env=args.env,
                                     eps_iters=args.eps_dur,
                                     eps_runtime=args.eps_runtime,
                                     eps_patience=args.eps_patience,
                                     random_seed=args.seed,
                                     k_prev_actions=args.k_prev_actions)

    env, eval_env = get_and_set_benchmarks(env, eval_env, args.datasets, args.test_datasets, args.overlap)

    p.RESULTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent = init_agent(args.agent, env)
    agent = agent.learn(total_timesteps=args.timesteps,
                        eval_freq=args.eval_freq,
                        eval_env=eval_env,
                        n_eval_episodes=args.eval_dur,
                        tb_log_name=save_name,
                        eval_log_path=str(p.RESULTS_DIRC))
    save_agent(agent, save_name)

    view_results(move_eval_results(save_name))


def view_results(results_file: str) -> None:
    results_file = p.RESULTS_DIRC / results_file
    if not results_file.exists():
        raise ValueError(f'results file {results_file} doesn\'t exist')

    with np.load(results_file) as f:
        print(f'{f["results"].shape[-1]} trials per evaluation\n')
        for idx, timestep in enumerate(f['timesteps']):
            results = f["results"][idx]
            eps_length = f["ep_lengths"][idx]
            print(f'at timestep {timestep}')
            print(f'   rewards:             {results}')
            print(f'   avg rewards:         {np.average(results)}')
            print(f'   episode lengths:     {eps_length}')
            print(f'   avg episode lengths: {np.average(eps_length)}\n')


def main(args: ap.Namespace) -> None:
    if args.view_results is not None:
        view_results(args.view_results)
    elif any([getattr(args, x) is None for x in ('env', 'agent', 'timesteps', 'save_name')]):
        raise ValueError(f'require an environment, agent, timesteps, and save name')
    else:
        train_and_eval_agent(args)


if __name__ == '__main__':
    main(apr.parser_user_args())

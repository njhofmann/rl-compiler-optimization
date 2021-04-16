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
    for agent, init_func in ('ppo', sb.PPO), ('ac', sb.A2C), ('dqn', sb.DQN):
        if agent_type == agent:
            return init_func(policy='MlpPolicy', env=env, verbose=1)
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
                           eval_datasets: Optional[List[str]]) -> Tuple[gy.CompilerGymWrapper, gy.CompilerGymWrapper]:
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
    env = gy.CompilerGymWrapper(compiler_env=args.env,
                                eps_iters=args.eps_dur,
                                eps_runtime=args.eps_runtime,
                                eps_patience=args.eps_patience,
                                random_seed=args.seed)

    eval_env = gy.CompilerGymWrapper(compiler_env=args.env,
                                     eps_iters=args.eps_dur,
                                     eps_runtime=args.eps_runtime,
                                     eps_patience=args.eps_patience,
                                     random_seed=args.seed)

    eval_env, env = get_and_set_benchmarks(env, eval_env, args.datasets, args.test_datasets)

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
    if args.view_results is None:
        train_and_eval_agent(args)
    else:
        view_results(args.view_results)


if __name__ == '__main__':
    main(apr.parser_user_args())

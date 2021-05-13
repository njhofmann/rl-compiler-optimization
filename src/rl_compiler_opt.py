import argparse as ap
import random as r
from typing import List, Tuple, Optional

from src import paths as p, arg_parser as apr, util as u, envs as e
from src.agent import init_agent, save_agent
from src.envs.init_envs import init_envs
from src.results import move_eval_results, view_results


def get_and_set_benchmarks(train_env: e.CompilerGymWrapper, eval_env: e.CompilerGymWrapper, datasets: List[str],
                           eval_datasets: Optional[List[str]], overlap: bool) \
        -> Tuple[e.CompilerGymWrapper, e.CompilerGymWrapper]:
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


def train_and_eval_agent(
        # general info / global args
        random_seed: int,
        verbose: int,
        save_name: str,

        # env args
        env: str,
        timesteps: int,
        eval_dur: int,
        eval_freq: int,
        datasets: List[str],
        test_datasets: Optional[List[str]],
        dataset_overlap: bool,
        log_training: bool,
        eps_dur: Optional[int],
        eps_runtime: Optional[int],
        eps_patience: Optional[int],
        k_prev_actions: Optional[int],
        neutral_rewards: Optional[int],

        # agent args
        agent: str,
        network_arch: List[int],
        learning_rate: float,
        gamma: float,
        max_grad_norm: int,
        buffer_size: int,
        n_steps: int,
        batch_size: int,
        vf_coef: float,
        ent_coef: float,
        gae_lambda: float,
        gradient_steps: int,
        learning_starts: int,
        target_update_interval: int,
        train_freq: int
        ) -> None:
    u.set_random_seed(random_seed)

    train_env, eval_env = init_envs(env=env,
                                    save_name=save_name,
                                    log_training=log_training,
                                    eps_dur=eps_dur,
                                    eps_runtime=eps_runtime,
                                    eps_patience=eps_patience,
                                    random_seed=random_seed,
                                    k_prev_actions=k_prev_actions,
                                    neutral_rewards=neutral_rewards)

    train_env, eval_env = get_and_set_benchmarks(train_env, eval_env, datasets, test_datasets, dataset_overlap)

    p.RESULTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent = init_agent(agent,
                       train_env,
                       random_seed=random_seed,
                       verbose=verbose,
                       learning_rate=learning_rate,
                       gamma=gamma,
                       max_grad_norm=max_grad_norm,
                       buffer_size=buffer_size,
                       n_steps=n_steps,
                       batch_size=batch_size,
                       vf_coef=vf_coef,
                       ent_coef=ent_coef,
                       gae_lambda=gae_lambda,
                       gradient_steps=gradient_steps,
                       learning_starts=learning_starts,
                       target_update_interval=target_update_interval,
                       train_freq=train_freq,
                       network_arch=network_arch)
    agent = agent.learn(total_timesteps=timesteps,
                        eval_freq=eval_freq,
                        eval_env=eval_env,
                        n_eval_episodes=eval_dur,
                        tb_log_name=save_name,
                        eval_log_path=str(p.RESULTS_DIRC))
    save_agent(agent, save_name)

    view_results(str(move_eval_results(save_name)))


def main(args: ap.Namespace) -> None:
    if args.view_results is not None:
        view_results(args.view_results)
    elif any([getattr(args, x) is None for x in ('env', 'agent', 'timesteps', 'save_name')]):
        raise ValueError(f'require an environment, agent, timesteps, and save name')
    else:
        train_and_eval_agent(timesteps=args.timesteps,
                             verbose=args.verbose,
                             agent=args.agent,
                             save_name=args.save_name,
                             env=args.env,
                             eval_freq=args.eval_freq,
                             eval_dur=args.eval_dur,
                             eps_runtime=args.eps_runtime,
                             eps_dur=args.eps_dur,
                             eps_patience=args.eps_patience,
                             log_training=args.log_training,
                             datasets=args.datasets,
                             test_datasets=args.test_datasets,
                             dataset_overlap=args.overlap,
                             random_seed=args.seed,
                             k_prev_actions=args.k_prev_actions,
                             neutral_rewards=args.neutral_rewards,
                             learning_rate=args.learning_rate,
                             buffer_size=args.buffer_size,
                             batch_size=args.batch_size,
                             n_steps=args.n_steps,
                             network_arch=args.network_arch,
                             max_grad_norm=args.max_grad_norm,
                             vf_coef=args.vf_coef,
                             ent_coef=args.ent_coef,
                             train_freq=args.train_freq,
                             gae_lambda=args.gae_lambda,
                             gradient_steps=args.gradient_steps,
                             gamma=args.gamma,
                             target_update_interval=args.target_update_interval,
                             learning_starts=args.learning_starts)


if __name__ == '__main__':
    main(apr.parser_user_args())

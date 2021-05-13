import argparse as ap
from typing import Tuple, List, Optional

import ray.tune as t
import ray.tune.suggest.bayesopt as b

import src.rl_compiler_opt as rl
import src.results as r
import src.paths as p
import pickle


def init_parser() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    parser.add_argument('--agent', '-a', choices=['dqn', 'a2c', 'ppo'])
    parser.add_argument('--random_starts', '-rs', default=5, type=int)
    parser.add_argument('--trials', '-t', default=30, type=int)
    parser.add_argument('--seed', '-s', default=492, type=int)
    return parser


def uniform_to_discrete(rand_value: Optional[float], discrete_vals: List[float]) -> Optional[float]:
    if rand_value is None:
        return None
    return discrete_vals[int(rand_value)]


def trainable(hyperparams: dict, agent: str, random_seed: int):
    t.utils.wait_for_gpu()

    save_name = f'{agent}_0'

    batch_size = uniform_to_discrete(hyperparams.get('batch_size'), [8, 16, 32])
    n_steps = uniform_to_discrete(hyperparams.get('n_steps'), [3, 4, 5, 6, 7, 8])
    buffer_size = uniform_to_discrete(hyperparams.get('buffer_size'), [1000, 4000, 7000, 10000])
    train_freq = uniform_to_discrete(hyperparams.get('train_freq'), [4, 8, 16])
    gradient_steps = uniform_to_discrete(hyperparams.get('gradient_steps'), [-1, 1, 4, 8])
    target_update_interval = uniform_to_discrete(hyperparams.get('target_update_interval'), [10, 100, 1000])
    rl.train_and_eval_agent(agent=agent,
                            env='llvm-autophase-ic-v0',
                            verbose=1,
                            save_name=save_name,
                            timesteps=2000000,
                            eval_dur=10,
                            eval_freq=1000000,
                            datasets=['blas-v0'],
                            log_training=False,
                            learning_starts=1000,
                            random_seed=random_seed,
                            dataset_overlap=False,
                            test_datasets=None,
                            eps_dur=1000,
                            eps_patience=None,
                            eps_runtime=None,
                            k_prev_actions=0,
                            neutral_rewards=0,
                            network_arch=[64, 64],
                            max_grad_norm=hyperparams['max_grad_norm'],
                            gamma=hyperparams['gamma'],
                            learning_rate=hyperparams['learning_rate'],
                            vf_coef=hyperparams.get('vf_coef'),
                            ent_coef=hyperparams.get('ent_coef'),
                            gae_lambda=hyperparams.get('gae_lambda'),
                            gradient_steps=gradient_steps,
                            target_update_interval=target_update_interval,
                            train_freq=train_freq,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            n_steps=n_steps)

    eval_results_file = f'{save_name}_eval_results.npz'
    t.report(last_avg_eval_reward=r.get_last_results(eval_results_file))


def get_tune_config(agent: str) -> Tuple[dict, List[dict]]:
    global_options = {'learning_rate': t.uniform(1e-5, 1e-3),
                      'max_grad_norm': t.uniform(.3, 10),
                      'gamma': t.uniform(.9, .999)}

    n_steps = t.uniform(0, 6)  # [3, 4, 5, 6, 7, 8]
    batch_size = t.uniform(0, 3)  # 0: 8, 1: 16, 2: 32
    ent_coef = t.uniform(1e-6, 0.1)
    vf_coef = t.uniform(.2, .6)
    gae_lambda = t.uniform(.8, .999)

    if agent == 'dqn':
        param_ranges = {'batch_size': batch_size,
                        'buffer_size': t.uniform(0, 4),  # 1000, 4000, 7000 10000,
                        'train_freq': t.uniform(0, 3),  # 4, 8, 16
                        'gradient_steps': t.uniform(0, 4),  # -1, 1, 4, 8
                        'target_update_interval': t.uniform(0, 3)}  # 10, 100, 1000
        priors = [{'learning_rate': 0.0005,
                   'max_grad_norm': 10,
                   'gamma': .99,
                   'batch_size': 0.0,  # 8
                   'buffer_size': 1.0,  # 1000, 4000, 7000 10000,
                   'train_freq': 0.0,  # 4, 8, 16
                   'gradient_steps': -1,  # -1, 1, 4, 8
                   'target_update_interval': 0.0}]  # 10
    elif agent == 'a2c':
        param_ranges = {'n_steps': n_steps,
                        'gae_lambda': gae_lambda,
                        'ent_coef': ent_coef,
                        'vf_coef': vf_coef}
        priors = [{'learning_rate': 0.0001,
                   'max_grad_norm': 10,
                   'gamma': .99,
                   'n_steps': 2.0,  # 5
                   'gae_lambda': 1.0,
                   'ent_coef': 0.01,
                   'vf_coef': 0.5}]
    elif agent == 'ppo':
        param_ranges = {'batch_size': batch_size,
                        'n_steps': n_steps,
                        'gae_lambda': gae_lambda,
                        'ent_coef': ent_coef,
                        'vf_coef': vf_coef}
        priors = [{'learning_rate': 0.0001,
                   'max_grad_norm': 0.5,
                   'gamma': .99,
                   'batch_size': 0,  # 8
                   'n_steps': 1.0,  # 4
                   'gae_lambda': 0.95,
                   'ent_coef': 0.0,
                   'vf_coef': .5}]
    else:
        raise ValueError(f'{agent} is not a supported type')
    return {**param_ranges, **global_options}, priors


def save_experiment_results(results, save_name: str):
    with open(p.HYPERPARAM_RESULTS_DIRC / f'{save_name}.pyk', 'wb') as f:
        pickle.dumps(results, f)


if __name__ == '__main__':
    args = init_parser().parse_args()
    config, priors = get_tune_config(args.agent)
    results = t.run(t.with_parameters(trainable, agent=args.agent, random_seed=args.seed),
                    config=config,
                    num_samples=args.trials,
                    metric='last_avg_eval_reward',
                    mode='max',
                    search_alg=b.BayesOptSearch(metric='last_avg_eval_reward',
                                                mode='max',
                                                random_search_steps=args.random_starts,
                                                points_to_evaluate=priors),
                    resources_per_trial={'gpu': 1,
                                         'cpu': 4})
    print(results.best_config)
    save_experiment_results(results, f'{args.agent}_hyperparam_results')

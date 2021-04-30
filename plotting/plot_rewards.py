import matplotlib.pyplot as plt
import pathlib as pl
import src.paths as p
from typing import Dict, Tuple, List
import numpy as np
import functools as ft


"""Plots the timestep vs average evaluation reward for some set of results"""


def moving_avg(x: np.ndarray, window: int):
    return np.convolve(x, np.ones(window), 'valid') / window


def plot_rewards(data: List[Tuple[str, str, str, np.ndarray, np.ndarray]], title: str) -> None:
    for (label, color, line_style, x, y) in data:
        x = np.concatenate([np.array([0]), x])
        y = np.concatenate([np.array([0]), y])
        plt.plot(x, y, color=color, label=label, linestyle=line_style)
    plt.title(title)
    plt.legend()
    plt.xlabel('Timestep')
    plt.ylabel(f'Mean Evaluation Reward')

    p.PLOTS_DIRC.mkdir(exist_ok=True, parents=True)
    plt.savefig(p.PLOTS_DIRC / f"{'_'.join([x.lower() for x in title.split(' ')])}.png")
    plt.close()


def pad_data(data: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    max_data_idx = np.argmax([len(x['timesteps']) for x in data])
    max_data = data[max_data_idx]
    for i, item in enumerate(data):
        if i != max_data_idx:
            item_len = len(item['timesteps'])
            item['timesteps'] = np.concatenate([item['timesteps'], max_data['timesteps'][item_len:]])
            item['results'] = np.concatenate([item['results'], max_data['results'][item_len:]])
            data[i] = item
    return data


def process_raw_files(files: List[str], window_size: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    data = [dict(np.load(p.RESULTS_DIRC / f'{x}.npz')) for x in files]

    lens = [len(x['timesteps']) for x in data]
    if len(set(lens)) != 1:
        data = pad_data(data)

    y = np.mean([np.mean(x['results'], axis=1) for x in data], axis=0)
    x = data[0]['timesteps']

    y = moving_avg(y, window_size)
    x = x[window_size-1:]
    return x, y


if __name__ == '__main__':
    # 1000 iterations
    a2c_1000_iters = ['a2c_blas-v0_1_eval_results', 'a2c_blas-v0_2_eval_results', 'a2c_blas-v0_3_eval_results']
    dqn_1000_iters = ['dqn_blas-v0_1_eval_results', 'dqn_blas-v0_2_eval_results', 'dqn_blas-v0_3_eval_results']
    ppo_1000_iters = ['ppo_blas-v0_1_eval_results', 'ppo_blas-v0_2_eval_results', 'ppo_blas-v0_3_eval_results']

    # 100 iterations
    a2c_100_iters = ['a2c_blas-v0_100_1_eval_results', 'a2c_blas-v0_100_2_eval_results', 'a2c_blas-v0_100_3_eval_results']
    dqn_100_iters = ['dqn_blas-v0_100_1_eval_results', 'dqn_blas-v0_100_2_eval_results', 'dqn_blas-v0_100_3_eval_results']
    ppo_100_iters = ['ppo_blas-v0_100_1_eval_results', 'ppo_blas-v0_100_2_eval_results', 'ppo_blas-v0_100_3_eval_results']

    # 50 patience
    a2c_50_pat = ['a2c_blas-v0_50_pat_1_eval_results', 'a2c_blas-v0_100_pat_2_eval_results', 'a2c_blas-v0_50_pat_3_eval_results']
    dqn_50_pat = ['dqn_blas-v0_50_pat_3_eval_results'] # 'dqn_blas-v0_50_pat_1_eval_results', 'dqn_blas-v0_100_pat_2_eval_results',
    ppo_50_pat = ['ppo_blas-v0_50_pat_1_eval_results', 'ppo_blas-v0_100_pat_2_eval_results', 'ppo_blas-v0_50_pat_3_eval_results']

    # 100 patience
    a2c_100_pat = ['a2c_blas-v0_100_pat_1_eval_results', 'a2c_blas-v0_100_pat_2_eval_results', 'a2c_blas-v0_100_pat_3_eval_results']
    dqn_100_pat = ['dqn_blas-v0_100_pat_1_eval_results', 'dqn_blas-v0_100_pat_2_eval_results', 'dqn_blas-v0_100_pat_3_eval_results']
    ppo_100_pat = ['ppo_blas-v0_100_pat_1_eval_results', 'ppo_blas-v0_100_pat_2_eval_results', 'ppo_blas-v0_100_pat_3_eval_results']

    a2c_data = [('100 Iterations', 'red', 'solid', *process_raw_files(a2c_100_iters)),
                ('1000 Iterations', 'blue', 'solid', *process_raw_files(a2c_1000_iters)),
                ('50 Patience', 'green', 'solid', *process_raw_files(a2c_50_pat)),
                ('100 Patience', 'orange', 'solid', *process_raw_files(a2c_100_pat))]

    dqn_data = [('100 Iterations', 'red', 'solid', *process_raw_files(dqn_100_iters)),
                ('1000 Iterations', 'blue', 'solid', *process_raw_files(dqn_1000_iters)),
                ('50 Patience', 'green', 'solid', *process_raw_files(dqn_50_pat)),
                ('100 Patience', 'orange', 'solid', *process_raw_files(dqn_100_pat))]

    ppo_data = [('100 Iterations', 'red', 'solid', *process_raw_files(ppo_100_iters)),
                ('1000 Iterations', 'blue', 'solid', *process_raw_files(ppo_1000_iters)),
                ('50 Patience', 'green', 'solid', *process_raw_files(ppo_50_pat)),
                ('100 Patience', 'orange', 'solid', *process_raw_files(ppo_100_pat))]

    for name, data in ('a2c', a2c_data), ('dqn', dqn_data), ('ppo', ppo_data):
        for (t, _, _, x, y) in data:
            print(name, t)
            print(x[::20])
            print(y[::20])
        plot_rewards(data, f'{name.upper()} Results')

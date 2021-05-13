import pathlib as pl

import numpy as np

from src import paths as p


def move_eval_results(model_name: str) -> pl.Path:
    old_file = create_results_path('evaluations.npz')
    new_file = p.RESULTS_DIRC / f'{model_name}_eval_results.npz'
    old_file.rename(new_file)
    return new_file


def create_results_path(results_file: str) -> pl.Path:
    results_file = p.RESULTS_DIRC / results_file
    if not results_file.exists():
        raise ValueError(f'results file {results_file} doesn\'t exist')
    return results_file


def get_last_results(results_file: str) -> float:
    results_path = create_results_path(results_file)
    with np.load(results_path) as f:
        return np.average(f['results'][-1])


def view_results(results_file: str) -> None:
    results_file = create_results_path(results_file)
    with np.load(results_file) as f:
        print(f'{f["results"].shape[-1]} trials per evaluation\n')
        for idx, timestep in enumerate(f['timesteps']):
            results = f['results'][idx]
            eps_length = f['eps_lengths'][idx]
            print(f'at timestep {timestep}')
            print(f'   rewards:             {results}')
            print(f'   avg rewards:         {np.average(results)}')
            print(f'   episode lengths:     {eps_length}')
            print(f'   avg episode lengths: {np.average(eps_length)}\n')


if __name__ == '__main__':
    print(get_last_results('a2c_blas-v0_1_eval_results.npz'))
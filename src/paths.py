import pathlib as pl

ROOT_DIRC = pl.Path(__file__).parent.parent
RESULTS_DIRC = ROOT_DIRC / 'results'
AGENTS_DIRC = ROOT_DIRC / 'agents'
TRAIN_LOGS = ROOT_DIRC / 'train_logs'
PLOTS_DIRC = ROOT_DIRC / 'plots'
HYPERPARAM_RESULTS_DIRC = ROOT_DIRC / 'hyperparam_results'

if __name__ == '__main__':
    print(type(RESULTS_DIRC))

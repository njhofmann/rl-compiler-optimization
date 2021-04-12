import pathlib as pl

ROOT_DIRC = pl.Path(__file__).parent.parent
RESULTS_DIRC = ROOT_DIRC / 'results'
AGENTS_DIRC = ROOT_DIRC / 'agents'

if __name__ == '__main__':
    print(type(RESULTS_DIRC))

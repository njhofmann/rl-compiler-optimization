import argparse as ap


def parser_user_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, choices=['llvm-autophase-ic-v0', 'llvm-autophase-codesize-v0'],
                        required=True, help='environment to learn a policy for')
    parser.add_argument('--agent', '-a', type=str, choices=['dqn', 'ac', 'ppo'], required=True,
                        help='agent to learn a policy with')
    parser.add_argument('--timesteps', '-t', type=int, required=True, help='number of timesteps to train an agent')
    parser.add_argument('--eval_freq', '-ef', type=int, default=1000,
                        help='evaluate the policy every X training timesteps')
    parser.add_argument('--eval_dur', '-vd', type=int, default=10,
                        help='how many episodes to evaluate the policy on during each testing session')
    parser.add_argument('--hold_out_count', '--hc', type=int, default=10,
                        help='how many states to randomly select before training to track their estimated values')
    parser.add_argument('--eps_dur', '-ed', type=int, default=None,
                        help='how many iterations to let each episode run at max')
    parser.add_argument('--eps_patience', '-ep', type=int, default=None,
                        help='max amount of "patience" to give during each episode')
    parser.add_argument('--eps_runtime', '-er', type=int, default=None,
                        help='amount of runtime to give each episode, in seconds')
    parser.add_argument('--save_name', '-sn', required=True, help='name to give to saved model and its results')
    parser.add_argument('--seed', '-s', type=int, default=None, help='random seed for pseudo random operations')
    parser.add_argument('--datasets', '-d', nargs='+', default=['github-v0'],
                        help='selection of datasets to train and eval an agent on, may overlap with testing benchmarks')
    parser.add_argument('--test_datasets', '-td',
                        help='selection of datasets to eval an agent on, if overlaps with training datasets - splits'
                             'the benchmarks in it dataset')
    parser.add_argument('--view_results', default=None,
                        help='view the results from a prior run, stored under the results file')
    return parser.parse_args()

import argparse as ap
from typing import List


def add_dqn_args(parser: ap.ArgumentParser) -> ap.ArgumentParser:
    """Adds arguments to the given parser for DQN agent specific hyperparameters"""
    parser.add_argument('--gradient_steps', type=int, default=4,
                        help='how many gradient steps to do after a training rollout, for DQN only')
    parser.add_argument('--train_freq', type=int, default=4,
                        help='updates the model every X steps')
    parser.add_argument('--learning_starts', type=int, default=1000,
                        help='how many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--target_update_interval', type=int, default=10000,
                        help='update target network every X environment steps')
    parser.add_argument('--buffer_size', '-us', type=int, default=10000,
                        help='buffer size to store previous environment experiences')
    return parser


def add_hyperparam_args(parser: ap.ArgumentParser) -> ap.ArgumentParser:
    """Adds arguments to the given parser for agent hyperparameters"""
    parser.add_argument('--network_arch', '-na', type=int, default=[64, 64], nargs='+',
                        help='architecture of each network in an agent')
    parser.add_argument('--learning_rate', '-lr', type=float, default=.99, help='learning rate of agent')
    parser.add_argument('--max_grad_norm', '-mgn', type=float, default=40, help='max value of gradient clipping')
    parser.add_argument('--gamma', '-g', type=float, default=.99, help='discount factor for future rewards')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        help='minibatch size of agent network updates, only for PPO and DQN')
    parser.add_argument('--n_steps', '-ns', type=int, default=5,
                        help='number of steps to run for each environment per update, only for A2C and PPO')
    parser.add_argument('--gae_lambda', type=float, default=1,
                        help='factor for trade-off of bias vs variance for Generalized Advantage Estimator Equivalent, '
                             'only for A2C and PPO')
    parser.add_argument('--ent_coef', type=float, default=0, help='entropy coefficient for loss, only for A2C and PPO')
    parser.add_argument('--vf_coef', type=float, default=.5, help='value coefficient for loss, only for A2C and PPO')

    parser = add_dqn_args(parser)
    return parser


def parser_user_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, choices=['llvm-autophase-ic-v0', 'llvm-autophase-codesize-v0'],
                        help='environment to learn a policy for')
    parser.add_argument('--agent', '-a', type=str, choices=['dqn', 'a2c', 'ppo'],
                        help='agent to learn a policy with')
    parser.add_argument('--timesteps', '-t', type=int, help='number of timesteps to train an agent')
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
    parser.add_argument('--save_name', '-sn', help='name to give to saved model and its results')
    parser.add_argument('--seed', '-s', type=int, default=None, help='random seed for pseudo random operations')
    parser.add_argument('--datasets', '-d', nargs='+', default=['github-v0'],
                        help='selection of datasets to train and eval an agent on, may overlap with testing benchmarks')
    parser.add_argument('--test_datasets', '-td', default=None,
                        help='selection of datasets to eval an agent on, may overlap with training benchmarks, if None '
                             'uses training benchmarks')
    parser.add_argument('--overlap', '-o', action='store_true',
                        help='if overlap between training and testing datasets, splits them')
    parser.add_argument('--view_results', default=None,
                        help='view the results from a prior run, stored under the results file')
    parser.add_argument('--log_training', '-lt', action='store_true', help='log training progress')
    parser.add_argument('--k_prev_actions', '-ka', type=int, default=0,
                        help='adds the k previous actions to the environment')
    parser.add_argument('--neutral_rewards', '-nr', type=float,
                        help='applies a negative reward to every "neutral action" - ie those that don\'t induce any '
                             'change to the environment')
    parser.add_argument('--verbose', '-v', default=1, type=int, help='verbosity of agents and environment')
    parser = add_hyperparam_args(parser)
    return parser.parse_args()

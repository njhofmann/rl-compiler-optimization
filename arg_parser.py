import argparse as ap
import compiler_gym as cg


def parser_user_args() -> ap.Namespace:
    parser = ap.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, choices=cg.COMPILER_GYM_ENVS, help='environment to learn a policy for')
    parser.add_argument('--agent', '-a', type=str, choices=['dqn', 'ac', 'ppo'], help='agent to learn a policy with')
    parser.add_argument('--eval_freq', '-ef', help='evaluate the policy every X training episodes')
    parser.add_argument('--eval_dur', '-vd', help='how many episodes to evaluate the policy on during testing')
    parser.add_argument('--hold_out_count', '--hc',
                        help='how many states to randomly select before training to track their estimated values')
    parser.add_argument('--eps_dur', '-ed', type=int, default=None,
                        help='how many iterations to let each episode run at max')
    parser.add_argument('--eps_patience', '-ep', type=int, default=None,
                        help='max amount of "patience" to give during each episode')
    parser.add_argument('--eps_runtime', '-er', type=int, default=None,
                        help='amount of runtime to give each episode, in seconds')
    return parser.parse_args()

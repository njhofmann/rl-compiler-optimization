import stable_baselines3 as sb
import compiler_gym as cg
import gym as g
import arg_parser as apr
import gym_wrapper as gy
import argparse as ap


def init_agent(agent_type: str, env: g.Env):
    if agent_type == 'ppo':
        return sb.PPO('mlp', env)
    elif agent_type == 'ac':
        return sb.A2C('mlp', env)
    return sb.DQN('mlp', env)


def main(args: ap.Namespace) -> None:
    env = gy.CompilerGymWrapper(compiler_env=g.make(args.env),
                                eps_iters=args.eps_dur,
                                eps_runtime=args.eps_runtime,
                                eps_patience=args.eps_patience)
    agent = init_agent(args.agent, env)


if __name__ == '__main__':
    main(apr.parser_user_args())

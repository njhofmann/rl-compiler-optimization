import stable_baselines3 as sb
import compiler_gym as cg
import gym as g
import argparse as ap
from typing import Union
import pathlib as pl
from src import paths as p, arg_parser as apr, gym_wrapper as gy

Agent = Union[sb.DQN, sb.A2C, sb.PPO]


def init_agent(agent_type: str, env: g.Env) -> Agent:
    for agent, init_func in ('ppo', sb.PPO), ('ac', sb.A2C), ('dqn', sb.DQN):
        if agent_type == agent:
            return init_func(policy='MlpPolicy', env=env, verbose=1)
    raise ValueError(f'agent {agent_type} is not a supported agent')


def save_agent(agent: Agent, name: str) -> None:
    p.AGENTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent.save(p.AGENTS_DIRC / f'{name}.zip')


def main(args: ap.Namespace) -> None:
    save_name = args.save_name
    env = gy.CompilerGymWrapper(compiler_env=args.env,
                                eps_iters=args.eps_dur,
                                eps_runtime=args.eps_runtime,
                                eps_patience=args.eps_patience,
                                random_seed=args.seed)
    eval_env = gy.CompilerGymWrapper(compiler_env=args.env,
                                     eps_iters=args.eps_dur,
                                     eps_runtime=args.eps_runtime,
                                     eps_patience=args.eps_patience,
                                     random_seed=args.seed)

    p.RESULTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent = init_agent(args.agent, env)
    agent = agent.learn(total_timesteps=args.timesteps,
                        eval_freq=args.eval_freq,
                        eval_env=eval_env,
                        n_eval_episodes=args.eval_dur,
                        tb_log_name=save_name,
                        eval_log_path=str(p.RESULTS_DIRC))
    save_agent(agent, save_name)


if __name__ == '__main__':
    main(apr.parser_user_args())

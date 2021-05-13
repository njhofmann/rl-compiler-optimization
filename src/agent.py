from typing import Union, List

import gym as g
import stable_baselines3 as sb

from src import paths as p

Agent = Union[sb.DQN, sb.PPO, sb.A2C]


def init_agent(agent_type: str,
               env: g.Env,
               verbose: int,
               random_seed: int,
               learning_rate: float,
               max_grad_norm: int,
               gamma: float,
               buffer_size: int,
               n_steps: int,
               batch_size: int,
               vf_coef: float,
               ent_coef: float,
               gae_lambda: float,
               gradient_steps: int,
               learning_starts: int,
               target_update_interval: int,
               train_freq: int,
               network_arch: List[int],
               ) -> Agent:
    policy_kwargs = {'net_arch': network_arch}
    if agent_type == 'dqn':
        return sb.DQN(policy='MlpPolicy',
                      env=env,
                      seed=random_seed,
                      verbose=verbose,
                      learning_rate=learning_rate,  # 5e-4,
                      max_grad_norm=max_grad_norm,  # 40,
                      gamma=gamma,
                      train_freq=train_freq,
                      batch_size=batch_size,  # 32,
                      buffer_size=buffer_size,  # 2000,
                      learning_starts=learning_starts,  # 1000,
                      gradient_steps=gradient_steps,  # 4,
                      target_update_interval=target_update_interval,  # 10,
                      policy_kwargs=policy_kwargs)
    elif agent_type == 'a2c':
        return sb.A2C(policy='MlpPolicy',
                      env=env,
                      verbose=verbose,
                      seed=random_seed,
                      learning_rate=learning_rate,  # .0001,
                      max_grad_norm=max_grad_norm,  # 40,
                      gamma=gamma,
                      n_steps=n_steps,
                      vf_coef=vf_coef,  # .5,
                      ent_coef=ent_coef,  # .01,
                      gae_lambda=gae_lambda,  # 1.0,
                      policy_kwargs=policy_kwargs)
    elif agent_type == 'ppo':
        return sb.PPO(policy='MlpPolicy',
                      env=env,
                      verbose=verbose,
                      seed=random_seed,
                      learning_rate=learning_rate,
                      max_grad_norm=max_grad_norm,
                      gamma=gamma,
                      batch_size=batch_size,
                      n_steps=n_steps,
                      gae_lambda=gae_lambda,
                      vf_coef=vf_coef,
                      ent_coef=ent_coef,
                      policy_kwargs=policy_kwargs)

    raise ValueError(f'agent {agent_type} is not a supported agent')


def save_agent(agent: Agent, name: str) -> None:
    p.AGENTS_DIRC.mkdir(exist_ok=True, parents=True)
    agent.save(p.AGENTS_DIRC / f'{name}.zip')

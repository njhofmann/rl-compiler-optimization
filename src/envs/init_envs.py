from typing import Optional, Tuple

from src import paths as p
from src.envs import gym_wrapper as gy


def init_envs(env: str, save_name: str, log_training: bool, eps_dur: Optional[int], eps_runtime: Optional[int],
              eps_patience: Optional[int], random_seed: int, k_prev_actions: Optional[int],
              neutral_rewards: Optional[int]) -> Tuple[gy.CompilerGymWrapper, gy.CompilerGymWrapper]:
    train_log_path = None
    if log_training:
        train_log_path = p.TRAIN_LOGS / f'{save_name}_training.txt'

    train_env = gy.CompilerGymWrapper(compiler_env=env,
                                      eps_iters=eps_dur,
                                      eps_runtime=eps_runtime,
                                      eps_patience=eps_patience,
                                      random_seed=random_seed,
                                      logging_path=train_log_path,
                                      k_prev_actions=k_prev_actions,
                                      neutral_reward=neutral_rewards)

    eval_env = gy.CompilerGymWrapper(compiler_env=env,
                                     eps_iters=eps_dur,
                                     eps_runtime=eps_runtime,
                                     eps_patience=eps_patience,
                                     random_seed=random_seed,
                                     k_prev_actions=k_prev_actions,
                                     neutral_reward=neutral_rewards)

    return train_env, eval_env
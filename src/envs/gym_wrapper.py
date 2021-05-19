import collections as c
import logging as log
import logging.handlers as lh
import math as m
import pathlib as pl
import random as r
import time as t
from typing import Optional, List

import compiler_gym as cg
import gym as g
import numpy as np
from stable_baselines3.common.env_checker import check_env


class CompilerGymWrapper(g.Env):
    """Custom Gym environment to wrap around Compiler Gym environments because there are custom observations not
    supported by Stable Baselines (as of now). In addition, supports several different termination kinds of termination
    criteria as that concept is ill-defined for Compiler Gym environments"""

    def __init__(self, compiler_env: str,
                 random_seed: Optional[int] = None,
                 eps_iters: Optional[int] =
                 None, eps_patience: Optional[int] = None,
                 eps_runtime: Optional[int] = None,
                 logging_path: Optional[pl.Path] = None,
                 k_prev_actions: int = 0,
                 neutral_reward: float = 0) -> None:
        super(CompilerGymWrapper, self).__init__()

        # number of previous actions to feed to the environment
        self.k_prev_actions_cnt = k_prev_actions
        self.k_prev_actions = c.deque([0. for _ in range(self.k_prev_actions_cnt)], maxlen=self.k_prev_actions_cnt)

        # compiler gym environment to wrap around
        self.compiler_env = self._init_compiler_env(compiler_env)
        self.action_space = self.compiler_env.action_space
        self.reward_space = self.compiler_env.reward_space
        self.observation_space = self._init_observation_space()

        if all([x is None for x in (eps_iters, eps_patience)]):
            raise ValueError('need at least one termination criteria')

        self.compiler_env.seed(random_seed)

        self.benchmarks = None

        # reward to apply to "neutral" actions that don't change the environment (ie have natural reward of 0)
        # should be small negative number like -0.001
        self.neutral_reward = neutral_reward

        # episode termination criteria
        self._eps_iters = eps_iters
        self._eps_patience = eps_patience
        self._eps_runtime = eps_runtime

        # track current episode's progression towards termination
        self._cur_eps_iter = eps_iters
        self._cur_eps_patience = eps_patience
        self._cur_eps_runtime = eps_runtime
        self._cur_best_total_reward = -m.inf
        self._cur_total_reward = 0

        self.log = self._init_logger(logging_path)

        # how many episodes have been run?
        self._eps_count = 0
        self._step_count = 0

        self.reset()

    def _init_observation_space(self) -> g.Space:
        # remove feature that is always a constant after normalization
        # make space for the k prev actions
        old_space = self.compiler_env.observation_space.space
        new_size = old_space.shape[0] - 1 + self.k_prev_actions_cnt
        new_low = np.array([old_space.low[0] for _ in range(new_size)])
        new_high = np.array([old_space.high[0] for _ in range(new_size)])
        return g.spaces.Box(low=new_low, high=new_high, shape=(new_size,), dtype=np.float64)

    def _log_info(self, info: str) -> None:
        if self.log:
            self.log.info(info)

    def _init_logger(self, logging_path: Optional[pl.Path]) -> Optional[log.Logger]:
        if logging_path is None:
            return None
        logging_path.parent.mkdir(exist_ok=True, parents=True)
        logger = log.getLogger(__name__)
        logger.setLevel(log.INFO)
        writer = lh.WatchedFileHandler(filename=logging_path)
        formatter = log.Formatter(log.BASIC_FORMAT)
        writer.setFormatter(formatter)
        logger.addHandler(writer)
        return logger

    def _init_compiler_env(self, env: str) -> cg.CompilerEnv:
        return g.make(env)

    def _add_prev_action(self, prev_action: int) -> None:
        # 0 is reserved for no prev action
        # standardize to help learning
        self.k_prev_actions.append((prev_action + 1) / len(self.action_space.flags))

    def _out_of_patience(self, reward: float) -> bool:
        if self._eps_patience is None:
            return False

        self._cur_total_reward += reward
        if self._cur_total_reward > self._cur_best_total_reward:
            self._cur_eps_patience = self._eps_patience
            self._cur_best_total_reward = self._cur_total_reward

        return self._cur_eps_patience == 0

    def _out_of_runtime(self) -> bool:
        return self._eps_runtime is not None and self._cur_eps_runtime < 0

    def _out_of_iterations(self) -> bool:
        return self._eps_iters is not None and self._cur_eps_iter == 0

    def _is_episode_over(self, reward: float) -> bool:
        return self._out_of_patience(reward) or self._out_of_runtime() or self._out_of_iterations()

    def _increment_eps(self, step_time: float) -> None:
        if self._eps_patience is not None:
            self._cur_eps_patience -= 1

        if self._eps_iters is not None:
            self._cur_eps_iter -= 1

        if self._eps_runtime is not None:
            self._cur_eps_runtime -= step_time

    def _reset_for_eps(self) -> None:
        self._log_info(f'episode ended, cumulative reward: '
                       f'{self._cur_total_reward}, '
                       f'best cumulative reward: {self._cur_best_total_reward}\n')
        self.reset()
        self._cur_eps_patience = self._eps_patience
        self._cur_eps_runtime = self._eps_runtime
        self._cur_eps_iter = self._eps_iters
        self._cur_best_total_reward = -m.inf
        self._cur_total_reward = 0

    def _get_cur_time(self) -> float:
        return t.time()

    def _normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        """Pseudo-normalizes an Autophase observation space by dividing each feature by the total number of instructions
        in the representation (item 51) and removing that feature"""
        return np.concatenate([observation[:51], observation[52:]]) / max(observation[51], 1)

    def _add_prev_actions_to_obs(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, np.array(self.k_prev_actions)])

    def _modify_obs(self, obervation: np.ndarray) -> np.ndarray:
        return self._add_prev_actions_to_obs(self._normalize_obs(obervation))

    def step(self, action):
        start_time = self._get_cur_time()
        obs, reward, done, info = self.compiler_env.step(action)
        elapsed_time = self._get_cur_time() - start_time

        self._add_prev_action(action)

        self._step_count += 1
        self._log_info(f'step {self._step_count}, reward: {reward}')

        self._increment_eps(elapsed_time)

        if not done:
            done = self._is_episode_over(reward)

        if done:
            self._reset_for_eps()
        elif not done and reward == 0.0 and self.neutral_reward is not None:
            reward += self.neutral_reward

        obs = self._modify_obs(obs)
        return obs, reward, done, info

    def _get_rand_benchmark(self) -> Optional[str]:
        return r.choice(self.benchmarks) if self.benchmarks else None

    def set_benchmarks(self, benchmarks: List[str]) -> None:
        self.benchmarks = benchmarks

    def reset(self):
        self._eps_count += 1
        self._step_count = 0
        benchmark = self._get_rand_benchmark()
        self._log_info(f'episode {self._eps_count}, benchmark {benchmark}')
        return self._modify_obs(self.compiler_env.reset(benchmark))

    def close(self):
        return self.compiler_env.close()

    def seed(self, seed=None):
        return self.compiler_env.seed(seed)

    def render(self, mode: str):
        return self.compiler_env.render(mode)

    def summary(self) -> None:
        print(f'Observation Space: {self.observation_space}')
        print(f'# of Actions: {self.action_space}')
        print(f'Reward Space: {self.reward_space}')


def test_env() -> None:
    env = CompilerGymWrapper('llvm-autophase-codesize-v0', eps_iters=1000, k_prev_actions=10)
    check_env(env)
    obs = env.reset()
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    env = CompilerGymWrapper('llvm-autophase-codesize-v0', eps_iters=1000, k_prev_actions=10)
    env.summary()
    test_env()

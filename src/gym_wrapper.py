import math as m
import time as t
from typing import Optional, List
import random as r
import pathlib as pl
import logging as log

import compiler_gym as cg
import gym as g
import numpy as np
from stable_baselines3.common.env_checker import check_env


class CompilerGymWrapper(g.Env):
    """Custom Gym environment to wrap around Compiler Gym environments because there are custom observations not
    supported by Stable Baselines (as of now). In addition, supports several different termination kinds of termination
    criteria as that concept is ill-defined for Compiler Gym environments"""

    def __init__(self, compiler_env: str, random_seed: Optional[int] = None,
                 eps_iters: Optional[int] = None, eps_patience: Optional[int] = None,
                 eps_runtime: Optional[int] = None, logging_path: Optional[pl.Path] = None) -> None:
        super(CompilerGymWrapper, self).__init__()
        # compiler gym environment to wrap around
        self.compiler_env = self._init_compiler_env(compiler_env)
        self.action_space = self.compiler_env.action_space
        self.reward_space = self.compiler_env.reward_space

        # remove feature that is always a constant after normalization
        old_space = self.compiler_env.observation_space.space
        self.observation_space = g.spaces.Box(low=old_space.low[:-1],
                                              high=old_space.high[:-1],
                                              shape=(old_space.shape[0] - 1,),
                                              dtype=np.float64)

        if all([x is None for x in (eps_iters, eps_patience)]):
            raise ValueError('need at least one termination criteria')

        self.compiler_env.seed(random_seed)

        self.benchmarks = None

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

        self.log = False
        if logging_path:
            logging_path.parent.mkdir(exist_ok=True, parents=True)
            log.basicConfig(filename=str(logging_path), level=log.INFO)
            self.log = True

        # how many episodes have been run?
        self._eps_count = 0
        self._step_count = 0

        self.reset()

    def _log_info(self, info: str) -> None:
        if self.log:
            log.info(info)

    def _init_compiler_env(self, env: str) -> cg.CompilerEnv:
        return g.make(env)

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
        # TODO true divide error...?
        return np.concatenate([observation[:51], observation[52:]]) / observation[51]

    def step(self, action):
        start_time = self._get_cur_time()
        obs, reward, done, info = self.compiler_env.step(action)
        elapsed_time = self._get_cur_time() - start_time

        self._step_count += 1
        self._log_info(f'step {self._step_count}, reward: {reward}')

        self._increment_eps(elapsed_time)

        if not done:
            done = self._is_episode_over(reward)

        if done:
            self._reset_for_eps()

        obs = self._normalize_obs(obs)
        return obs, reward, done, info

    def _get_rand_benchmark(self) -> Optional[str]:
        # TODO explain me
        if not self.benchmarks:
            return None
        return r.choice(self.benchmarks)

    def set_benchmarks(self, benchmarks: List[str]) -> None:
        self.benchmarks = benchmarks

    def reset(self):
        self._eps_count += 1
        self._step_count = 0
        benchmark = self._get_rand_benchmark()
        self._log_info(f'episode {self._eps_count}, benchmark {benchmark}')
        return self._normalize_obs(self.compiler_env.reset(benchmark))

    def close(self):
        return self.compiler_env.close()

    def seed(self, seed=None):
        return self.compiler_env.seed(seed)

    def render(self, mode: str):
        return self.compiler_env.render(mode)

    def summary(self) -> None:
        print(f'Observation Space: {self.observation_space}')
        print(f'Action Space: {self.action_space}')
        print(f'Reward Space: {self.reward_space}')
        print(f'Active Benchmarks: {self.compiler_env.benchmarks}')


def test_env() -> None:
    env = CompilerGymWrapper('llvm-autophase-codesize-v0', eps_iters=1000)
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
    env = CompilerGymWrapper('llvm-autophase-codesize-v0', eps_iters=1000)
    env.summary()

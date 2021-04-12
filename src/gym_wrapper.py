import gym as g
import compiler_gym as cg
import math as m
import time as t

from typing import Optional

from stable_baselines3.common.env_checker import check_env
import stable_baselines3.common.vec_env as v


class CompilerGymWrapper(g.Env):
    """Custom Gym environment to wrap around Compiler Gym environments because there are custom observations not
    supported by Stable Baselines (as of now). In addition, supports several different termination kinds of termination
    criteria as that concept is ill-defined for Compiler Gym environments"""

    def __init__(self, compiler_env: str, random_seed: Optional[int] = None, eps_iters: Optional[int] = None,
                 eps_patience: Optional[int] = None, eps_runtime: Optional[int] = None) -> None:
        super(CompilerGymWrapper, self).__init__()
        # compiler gym environment to wrap around
        self.compiler_env = self._init_compiler_env(compiler_env)
        self.action_space = self.compiler_env.action_space
        self.observation_space = self.compiler_env.observation_space.space

        if all([x is None for x in (eps_iters, eps_patience)]):
            raise ValueError('need at least one termination criteria')

        self.compiler_env.seed(random_seed)

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

        self.reset()

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
        self.reset()
        self._cur_eps_patience = self._eps_patience
        self._cur_eps_runtime = self._eps_runtime
        self._cur_eps_iter = self._eps_iters
        self._cur_best_total_reward = -m.inf
        self._cur_total_reward = 0

    def _get_cur_time(self) -> float:
        return t.time()

    def step(self, action):
        start_time = self._get_cur_time()
        obs, reward, done, info = self.compiler_env.step(action)
        elapsed_time = self._get_cur_time() - start_time

        self._increment_eps(elapsed_time)

        if not done:
            done = self._is_episode_over(reward)

        if done:
            self._reset_for_eps()

        return obs, reward, done, info

    def reset(self):
        return self.compiler_env.reset()

    def close(self):
        return self.compiler_env.close()

    def seed(self, seed=None):
        return self.compiler_env.seed(seed)

    def render(self, mode: str):
        return self.compiler_env.render(mode)


def test_env() -> None:
    env = CompilerGymWrapper('llvm-autophase-codesize-v0', eps_iters=1000)
    check_env(env)
    env = v.VecNormalize(v.DummyVecEnv([lambda: env]))
    obs = env.reset()
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    test_env()
    print()

import envpool
import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces import Box
import numpy as np

Array = np.ndarray


class MultiAgentWrapper(gym.Wrapper):
    def __init__(self, env: Env, envpool: bool = False):
        super().__init__(env)

        self.n_agents = 2
        self.single_observation_space = Box(-np.inf, np.inf, (17,), np.float64)
        self.single_action_space = Box(-1.0, 1.0, (3,), np.float32)
        self.envpool = envpool

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(
            self.action(actions)
        )
        return self.observation(observation), reward, terminated, truncated, info

    def action(self, actions: Array) -> Array:
        if self.envpool:
            action = np.concatenate([actions[:, 0], actions[:, 1]], axis=-1)
        else:
            action = np.concatenate([actions[0], actions[1]], axis=-1)
        action = np.array(action, np.float64)
        return np.clip(action, -1.0, 1.0)

    def observation(self, observation):
        ind1 = np.array(
            [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 5, 6, 7, 14, 15, 16]
        )  # back then front
        ind2 = np.array(
            [0, 1, 5, 6, 7, 8, 9, 10, 14, 15, 16, 2, 3, 4, 11, 12, 13]
        )  # front then back
        if len(observation.shape) == 2:
            observation1 = observation[:, ind1]
            observation2 = observation[:, ind2]
        else:
            observation1 = observation[ind1]
            observation2 = observation[ind2]
        return np.stack([observation1, observation2], axis=-2)


def make_ma_half_cheetah():
    """Makes a single multi-agents Half Cheetah environment
    where wrappers have been applied"""

    env = gym.make("HalfCheetah-v4")
    env = MultiAgentWrapper(env)
    return env


def make_envpool_ma_half_cheetah(n_envs: int):
    """Makes a parallelized multi-agents Half Cheetah environment
    where wrappers have been applied"""

    envs = envpool.make("HalfCheetah-v4", env_type="gymnasium", num_envs=n_envs)
    envs = MultiAgentWrapper(envs, True)
    return envs

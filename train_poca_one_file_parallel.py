from collections import deque

from einops import rearrange
import envpool
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np

from poca_one_file_single_opt_parallel import POCA
from rl_tools.config import get_config

import wandb

# region MultiAgent env


def to_ma_obs(observation):
    # E single obs (E, 17)
    # -> E ma obs (E, 2, 17)
    ind1 = np.array(
        [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 5, 6, 7, 14, 15, 16]
    )  # back then front
    ind2 = np.array(
        [0, 1, 5, 6, 7, 8, 9, 10, 14, 15, 16, 2, 3, 4, 11, 12, 13]
    )  # front then back
    observation1 = observation[:, ind1]
    observation2 = observation[:, ind2]
    return np.stack([observation1, observation2], axis=1)


def to_single_action(actions):
    # E ma action (E, 2, 3)
    # -> E single action (E, 6)
    actions = np.array(actions, np.float64)
    return rearrange(actions, "e n s -> e (n s)")


# endregion

# region Buffer


class OnPolicyBuffer:
    def __init__(self) -> None:
        self.ma_observations = []
        self.ma_actions = []
        self.ma_log_probs = []
        self.rewards = []
        self.dones = []
        self.ma_next_observations = []

    def add(self, ma_obs, ma_action, ma_log_prob, reward, done, ma_next_obs):
        self.ma_observations.append(ma_obs)
        self.ma_actions.append(ma_action)
        self.ma_log_probs.append(ma_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.ma_next_observations.append(ma_next_obs)

    def sample(self):
        return {
            "observations": np.array(self.ma_observations),
            "actions": np.array(self.ma_actions),
            "log_probs": np.array(self.ma_log_probs),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "next_observations": np.array(self.ma_next_observations),
        }


# endregion


def get_trajectory(env: gym.Env, ma_agent: POCA, n_steps: int, last_ma_obs, logs):
    """
    env is an envpool env !
    """
    buffer = OnPolicyBuffer()

    ma_obs = last_ma_obs
    for t in range(n_steps):
        ma_actions, ma_log_probs, _, _ = ma_agent.get_action(ma_obs)
        next_obs, rewards, dones, truncs, infos = env.step(
            np.clip(to_single_action(ma_actions), -1, 1)
        )
        logs["tmp_episode_return"] += rewards

        next_ma_obs = to_ma_obs(next_obs)
        buffer.add(ma_obs, ma_actions, ma_log_probs, rewards, dones, next_ma_obs)

        ma_obs = next_ma_obs
        for i, (d, t) in enumerate(zip(dones, truncs)):
            if d or t:
                logs["ep_reward_deque"].append(logs["tmp_episode_return"][i])
                logs["tmp_episode_return"][i] = 0.0
                logs["ep_reward"] = sum(logs["ep_reward_deque"]) / len(
                    logs["ep_reward_deque"]
                )

    return buffer, ma_obs, logs


def main(use_wandb=False):
    config = get_config("./config.yaml")
    envs = envpool.make(
        "HalfCheetah-v4", "gymnasium", num_envs=config["n_envs"], seed=config["seed"]
    )

    config["n_agents"] = 2
    config["n_actions"] = 3  # hardcoded
    config["observation_space"] = Box(-np.inf, np.inf, (17,), np.float64)
    config["action_space"] = Box(-1.0, 1.0, (6,), np.float32)

    if use_wandb:
        wandb.init(entity="raffael", project="poca_halfcheetah", config=config)

    agent = POCA(config)

    n_trajs = (
        config["n_steps_in_environment"]
        // config["n_steps_trajectory"]
        // config["n_envs"]
    )
    obs, _ = envs.reset()
    ma_obs = to_ma_obs(obs)

    logs = {
        "step": 0,
        "ep_reward": 0.0,
        "ep_reward_deque": deque(maxlen=20),
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "tmp_episode_return": np.zeros((config["n_envs"],)),
    }
    for k in range(n_trajs):
        buffer, ma_obs, logs = get_trajectory(
            envs, agent, config["n_steps_trajectory"], ma_obs, logs
        )
        logs["step"] += config["n_steps_trajectory"] * config["n_envs"]
        logs.update(agent.improve(buffer))

        print(
            logs["step"],
            logs["ep_reward"],
            logs["actor_loss"],
            logs["baseline_loss"],
            logs["value_loss"],
            logs["entropy"],
        )
        if use_wandb:
            wandb.log(
                {
                    "step": logs["step"],
                    "ep_reward": logs["ep_reward"],
                    "actor_loss": logs["actor_loss"],
                    "baseline_loss": logs["baseline_loss"],
                    "value_loss": logs["value_loss"],
                    "entropy": logs["entropy"],
                }
            )


if __name__ == "__main__":
    main(False)

import gymnasium as gym
import numpy as np

from poca_one_file import POCA
from rl_tools.config import get_config

import wandb

# region MultiAgent env


def to_ma_obs(observation):
    # single obs (17,)
    # -> ma obs (2, 17)
    ind1 = np.array(
        [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 5, 6, 7, 14, 15, 16]
    )  # back then front
    ind2 = np.array(
        [0, 1, 5, 6, 7, 8, 9, 10, 14, 15, 16, 2, 3, 4, 11, 12, 13]
    )  # front then back
    observation1 = observation[ind1]
    observation2 = observation[ind2]
    return np.stack([observation1, observation2], axis=0)


def to_single_action(actions):
    # ma action (2, 3)
    # -> single action (6,)
    return np.reshape(actions, (6,))


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
    buffer = OnPolicyBuffer()

    ma_obs = last_ma_obs
    for t in range(n_steps):
        ma_action, ma_log_prob = ma_agent.get_action(ma_obs)
        next_obs, reward, done, trunc, info = env.step(to_single_action(ma_action))
        logs["tmp_ep_reward"] += reward

        next_ma_obs = to_ma_obs(next_obs)
        buffer.add(ma_obs, ma_action, ma_log_prob, reward, done, next_ma_obs)

        ma_obs = next_ma_obs
        if done or trunc:
            logs["last_ep_reward"] = logs["tmp_ep_reward"]
            logs["tmp_ep_reward"] = 0.0

            obs, _ = env.reset()
            ma_obs = to_ma_obs(obs)

    return buffer, ma_obs, logs


def main():
    config = get_config("./config.yaml")
    env = gym.make("HalfCheetah-v4")

    config["n_agents"] = 2
    config["n_actions"] = env.action_space.shape[0] // 2
    config["observation_space"] = env.observation_space
    config["action_space"] = env.action_space

    # wandb.init(entity="raffael", project="poca_halfcheetah", config=config)

    agent = POCA(config)

    n_trajs = config["n_steps_in_environment"] // config["n_steps_trajectory"]
    obs, _ = env.reset(seed=config["seed"])
    ma_obs = to_ma_obs(obs)

    logs = {
        "step": 0,
        "last_ep_reward": 0.0,
        "tmp_ep_reward": 0.0,
        "actor_loss": 0.0,
        "critic_loss": 0.0,
    }
    for k in range(n_trajs):
        buffer, ma_obs, logs = get_trajectory(
            env, agent, config["n_steps_trajectory"], ma_obs, logs
        )
        logs["step"] += config["n_steps_trajectory"]
        logs.update(agent.improve(buffer))

        print(
            logs["step"],
            logs["last_ep_reward"],
            logs["actor_loss"],
            logs["critic_loss"],
        )
        # wandb.log(logs)


if __name__ == "__main__":
    main()

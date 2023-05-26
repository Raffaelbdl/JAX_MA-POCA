from collections import deque

import numpy as np

from rltools.config import get_config
from rltools.update import avg

from poca.agent import POCA, collect_trajectory

from envs.ma_half_cheetah import make_envpool_ma_half_cheetah

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import wandb
from absl import app, flags

flags.DEFINE_boolean("use_wandb", False, "")
flags.DEFINE_string("entity", "entity", "")
flags.DEFINE_string("project", "project", "")
FLAGS = flags.FLAGS


def train(config: dict):
    envs = make_envpool_ma_half_cheetah(config["n_envs"])

    config["n_agents"] = envs.n_agents
    config["observation_space"] = envs.single_observation_space
    config["action_space"] = envs.single_action_space

    if FLAGS.use_wandb:
        wandb.init(entity=FLAGS.entity, project=FLAGS.project, config=config)

    agent = POCA(config)
    logs = {
        "step": 0,
        "tmp_episode_reward": np.zeros((config["n_envs"],)),
        "avg_episode_reward": deque(maxlen=20),
    }

    n_steps_iteration = config["n_envs"] * config["n_steps_per_trajectory"]
    n_trajectory_collections = config["n_env_steps"] // n_steps_iteration
    last_observations, _ = envs.reset()
    for i in range(n_trajectory_collections):
        buffer, last_observations, logs = collect_trajectory(
            envs, agent, config["n_steps_per_trajectory"], last_observations, logs
        )
        logs.update(agent.improve(buffer))
        logs["step"] += n_steps_iteration

        if len(logs["avg_episode_reward"]) > 0:
            print(logs["step"], avg(logs["avg_episode_reward"]))
            if FLAGS.use_wandb:
                wandb.log(
                    {
                        "step": logs["step"],
                        "policy_loss": logs["policy_loss"],
                        "approx_kl": logs["approx_kl"],
                        "entropy": logs["entropy"],
                        "value_loss": logs["value_loss"],
                        "baseline": logs["baseline"],
                        "episode_return": avg(logs["avg_episode_reward"]),
                    }
                )


def main(_):
    config = get_config("./configs/train_poca_config.yaml")
    train(config)


if __name__ == "__main__":
    app.run(main)

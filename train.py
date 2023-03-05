import pickle
from typing import List

import gym.spaces as spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import numpy as np

from poca.poca import POCA

import wandb


def wrap_observations(env_obs: dict):
    obs = []

    n_agents = len(env_obs.keys())
    for agent, value in env_obs.items():
        if isinstance(value, dict):
            if "observation" in value.keys():
                obs.append(value["observation"])
            else:
                raise NotImplementedError

        else:
            if not isinstance(value, list):
                value = [value]
            obs.append(value)

    if n_agents > 1:
        obs = zip(*obs)
        obs = [np.stack(o, axis=0) for o in obs]
    else:
        obs = [np.array(o)[None, ...] for o in obs]

    return obs  # [N, size]


def wrap_actions(actions: np.array, agents: List[str], live_agents: List[str]):
    if len(actions) > 1:
        return {
            agent: action
            for agent, action in zip(agents, actions)
            if agent in live_agents
        }
    elif len(live_agents) >= 1:
        return {live_agents[0]: actions[0][0]}
    return {}


def wrap_rewards(rewards: dict):
    reward = sum(rewards.values()) / len(rewards.values())
    return reward


def wrap_dones(dones: dict):
    done = any(dones.values())
    return done


def wrap_step(obs, rwds, dones, infos):
    obs = wrap_observations(obs)
    rwd = wrap_rewards(rwds)
    done = wrap_dones(dones)
    return obs, rwd, done, infos


def main():
    unity_env = UnityEnvironment(
        file_name="../Equilibrium/Equilibrium_lowmass.x86_64", no_graphics=False
    )
    env = UnityParallelEnv(unity_env)
    agents = env._live_agents

    observation_space = env.observation_spaces[env.agents[0]]
    if not isinstance(observation_space, spaces.Tuple):
        observation_space = spaces.Tuple([observation_space])

    CONFIG = {
        "seed": 0,
        "obs_shape": [o.shape for o in observation_space],
        "observation_embed_size": 128,
        "n_agents": len(env.agents),
        "n_actions": env.action_spaces[env.agents[0]].n,
        "lambda": 0.95,
        "discount": 0.99,
        "b_coef": 1.0,
        "v_coef": 0.5,
        "entropy_coef": 0.02,
        "buffer_capacity": 512,
        "n_epochs": 1,
        "n_minibatchs": 4,
        "learning_rate": 3e-4,
        "epsilon": 0.2,
        "save_frequency": 20000,
    }

    wandb.init(entity="raffael", project="poca")
    wandb.config = CONFIG
    poca = POCA(CONFIG)

    logs = {"steps": 0, "loss": 0.0}

    obs = wrap_observations(env.reset())
    episode_reward = 0
    for i in range(1, int(5e6) + 1):
        logs["steps"] = i

        actions, log_probs = poca.get_action(obs)
        env_actions = wrap_actions(actions, agents, env._live_agents)
        next_obs, reward, done, _ = wrap_step(*env.step(env_actions))
        poca.buffer.add(obs, actions, reward, done, next_obs, log_probs)

        episode_reward += reward

        if poca.improve_condition:
            logs = poca.improve(logs)

        obs = next_obs
        if done:
            obs = wrap_observations(env.reset())
            logs["episode_rewards"] = episode_reward
            episode_reward = 0
            print(logs)
            wandb.log(logs)

        if i % CONFIG["save_frequency"] == 0:
            pickle.dump(poca.params, open(f"params_{i}", "wb"))
            pickle.dump(poca.opt_state, open(f"optstate_{i}", "wb"))
            print(f"Saved at {i} steps")


if __name__ == "__main__":
    main()

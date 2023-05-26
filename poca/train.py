from rltools.buffer import OnPolicyBuffer
from poca.agent import POCA

# region Train


def collect_trajectory(envs, agent: POCA, n_steps_trajectory, last_observations, logs):
    """env is an envpool environment"""
    buffer = OnPolicyBuffer()
    observations = last_observations

    for t in range(n_steps_trajectory):
        actions, log_probs = agent.get_action(observations)
        # halfcheetah specific
        next_observations, rewards, dones, truncs, info = envs.step(
            actions[:, 0], actions[:, 1]
        )

        logs["tmp_episode_reward"] += rewards

        buffer.add(observations, actions, log_probs, rewards, dones, next_observations)
        observations = next_observations

        for i, (d, t) in enumerate(zip(dones, truncs)):
            if d or t:
                logs["avg_episode_reward"].append(logs["tmp_episode_reward"][i])
                logs["tmp_episode_reward"][i] = 0.0

    return buffer, next_observations, logs


# endregion

import jax.numpy as jnp


def make_baseline_inputs(observations, actions):
    new_obs = []
    new_actions = []
    new_obs_only = []

    n_agents = actions.shape[1]
    for i in range(n_agents):
        inds = list(range(n_agents))
        inds.pop(i)
        i_excluded = jnp.array(inds, jnp.int32)

        new_obs.append(observations[:, i_excluded])
        new_actions.append(actions[:, i_excluded])
        new_obs_only.append(observations[:, i : i + 1])
    # N * [B, N-1 / 1, size_O / size_A]
    new_obs = jnp.stack(new_obs, axis=1)
    new_actions = jnp.stack(new_actions, axis=1)
    new_obs_only = jnp.stack(new_obs_only, axis=1)
    # B, N, size_O / size_A
    return new_obs, new_actions, new_obs_only

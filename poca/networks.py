from typing import Callable, Optional

import distrax as dx
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

import rltools.networks as nets
from rltools.types import Array

from poca.baseline_input import make_baseline_inputs


class ContinuousPolicyNetwork(hk.Module):
    """Continuous Policy Network for environment with vector observations"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.n_actions = np.prod(np.array(config["action_space"].shape))

    def __call__(self, observations: Array) -> tuple[Array, Array]:
        x = nets.LinearEncoder([128, 128], "encoder")(observations)

        locs = hk.Linear(self.n_actions)(x)
        log_scales = hk.get_parameter(
            "log_scales", shape=(1, self.n_actions), dtype=jnp.float32, init=jnp.zeros
        )
        log_scales = jnp.broadcast_to(log_scales, locs.shape)

        return dx.Normal(locs, jnp.exp(log_scales))


class ResidualSelfAttention(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["embed_size"]

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs

        x_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x_attn = hk.MultiHeadAttention(
            num_heads=4,
            key_size=self.embed_size,
            model_size=self.embed_size,
            w_init=hk.initializers.VarianceScaling(),
        )(x_norm, x_norm, x_norm)
        x = x + x_attn

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        return x


class BaselineNetwork(hk.Module):
    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        embed_size = config["embed_size"]
        self.obs_encoder = nets.LinearEncoder(
            [embed_size, embed_size], "baseline_obs_encoder"
        )
        self.obs_actions_encoder = nets.LinearEncoder(
            [embed_size, embed_size], "baseline_action_obs_encoder"
        )
        self.rsa = ResidualSelfAttention(config, "baseline_rsa")

    def __call__(self, obs_only: Array, obs: Array, actions: Array):
        rsa_inputs = self.obs_encoder(obs_only)

        obs_actions = jnp.concatenate([obs, actions], axis=-1)
        embed_obs_actions = self.obs_actions_encoder(obs_actions)
        # N_agents axis
        rsa_inputs = jnp.concatenate([rsa_inputs, embed_obs_actions], axis=-2)

        # we assume that there are no nans in obs
        rsa_outputs = self.rsa(rsa_inputs)
        avg_outputs = jnp.mean(rsa_outputs, axis=(-2))

        # value head
        x = nets.LinearEncoder([256, 256], "baseline_head")(avg_outputs)

        return hk.Linear(1)(x)


class ValueNetwork(hk.Module):
    """Value module

    Inputs:
        a batch of observations (B, N, size)
    """

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        embed_size = config["embed_size"]
        self.obs_encoder = nets.LinearEncoder(
            [embed_size, embed_size], "value_obs_encoder"
        )
        self.rsa = ResidualSelfAttention(config, "value_rsa")

    def __call__(self, obs: jnp.ndarray):
        rsa_inputs = self.obs_encoder(obs)

        # we assume that there are no nans in obs
        rsa_outputs = self.rsa(rsa_inputs)
        avg_outputs = jnp.mean(rsa_outputs, axis=(-2))

        # value head
        x = nets.LinearEncoder([256, 256], "value_head")(avg_outputs)

        return hk.Linear(1)(x)


def get_networks_fn(config: dict):
    key1, key2, key3 = jrd.split(jrd.PRNGKey(config["seed"]), 3)
    obs_shape = config["observation_space"].shape
    action_shape = config["action_space"].shape
    n_agents = config["n_agents"]

    single_observation = jnp.zeros((1,) + obs_shape)
    observations = jnp.zeros((1, n_agents) + obs_shape)
    actions = jnp.zeros((1, n_agents) + action_shape)

    @hk.transform
    def policy_transformed(observations):
        return ContinuousPolicyNetwork(config, "policy")(observations)

    policy_fwd = policy_transformed.apply
    policy_params = policy_transformed.init(key1, single_observation)

    @hk.transform
    def baseline_transformed(observations, actions, obs_only):
        return BaselineNetwork(config, "baseline")(obs_only, observations, actions)

    d_obs, d_actions, d_obs_only = make_baseline_inputs(observations, actions)
    baseline_fwd = baseline_transformed.apply
    baseline_params = baseline_transformed.init(key2, d_obs_only, d_obs, d_actions)

    @hk.transform
    def value_transformed(observations):
        return ValueNetwork(config, "value")(observations)

    value_fwd = value_transformed.apply
    value_params = value_transformed.init(key3, observations)

    fwd = [policy_fwd, baseline_fwd, value_fwd]
    params = [policy_params, baseline_params, value_params]

    return fwd, params

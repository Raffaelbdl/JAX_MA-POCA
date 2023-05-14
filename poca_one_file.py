"""Training on MA-HalfCheetah"""
from copy import copy
from typing import List, Optional, Type

from einops import rearrange
import distrax as dx
import haiku as hk
import jax
import jax.lax as jlax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import optax

from rl_tools.update import update
from rlax import lambda_returns

# region Networks


class LinearEncoder(hk.Module):
    """Simple observation encoder"""

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)
        self.embed_size = config["embed_size"]

    def __call__(self, x: jnp.ndarray):
        x = jnn.relu(hk.Linear(self.embed_size)(x))
        return jnn.relu(hk.Linear(self.embed_size)(x))


class ResidualSelfAttention(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["embed_size"]

    def __call__(self, inputs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
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

        numerator = jnp.sum(x * mask, axis=1)
        denominator = jnp.sum(mask, axis=1) + 1e-8

        return numerator / denominator


class Actor(hk.Module):
    """Actor module

    Inputs:
        a batch of observations
    Outputs:
        a batch of actions and a batch of log_probs
        where log_probs is sum(axis=-1, keepdims=True)
    """

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        self.n_actions = config["n_actions"]
        self.obs_encoder = LinearEncoder(config, "obs_encoder")

        self.log_std_min = config["log_std_min"]
        self.log_std_max = config["log_std_max"]

    def __call__(self, obs: jnp.ndarray):
        x = self.obs_encoder(obs)

        locs = hk.Linear(self.n_actions)(x)

        log_stds = jnn.tanh(hk.Linear(self.n_actions)(x))
        log_stds = 0.5 * (self.log_std_max - self.log_std_min) * (log_stds + 1)
        log_stds += self.log_std_min

        return self.get_action(locs, log_stds)

    def get_action(self, locs, log_stds):
        stds = jnp.exp(log_stds)
        normal_dist = dx.Normal(locs, stds)
        samples, log_probs = normal_dist.sample_and_log_prob(seed=hk.next_rng_key())

        actions = jnn.tanh(samples)

        log_probs -= jnp.log(1 - jnp.square(actions) + 1e-6)
        log_probs = jnp.sum(log_probs, axis=-1, keepdims=True)

        return actions, log_probs


class Critic(hk.Module):
    """Critic module

    Inputs:
        a batch of observations (B, N, size)
        a batch of actions (B, N, size)
        in the case of calculating the baseline,
        we add a batch of observations_only, that are not appaired to actions
    """

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        self.obs_encoder = LinearEncoder(config, "obs_encoder")
        self.obs_actions_encoder = LinearEncoder(config, "action_obs_encoder")
        self.rsa = ResidualSelfAttention(config, "rsa")

    def __call__(
        self,
        obs_only: jnp.ndarray,
        obs: jnp.ndarray = None,
        actions: jnp.ndarray = None,
    ):
        rsa_inputs = self.obs_encoder(obs_only)

        if obs is not None:
            obs_actions = jnp.concatenate([obs, actions], axis=-1)
            embed_obs_actions = self.obs_actions_encoder(obs_actions)
            # N_agents axis
            rsa_inputs = jnp.concatenate([rsa_inputs, embed_obs_actions], axis=1)

        # we assume that there are no nans in obs
        rsa_mask = jnp.ones_like(rsa_inputs)
        rsa_outputs = self.rsa(rsa_inputs, rsa_mask)

        # value head
        x = jnn.relu(hk.Linear(256)(rsa_outputs))
        x = jnn.relu(hk.Linear(256)(x))
        return hk.Linear(1)(x)


def get_continuous_networks(config: dict, key: jrd.PRNGKeyArray):
    @hk.transform
    def actor_transformed(observations):
        return Actor(config, "actor")(observations)

    @hk.transform
    def critic_transformed(observations, actions, obs_only):
        return Critic(config, "critic")(observations, actions, obs_only)

    key1, key2 = jrd.split(key, 2)
    n_agents = config["n_agents"]
    obs_shape = config["observation_space"].shape
    action_shape = (config["n_actions"],)  # TODO

    dummy_obs = np.zeros((1, n_agents - 1) + obs_shape)
    dummy_actions = np.zeros((1, n_agents - 1) + action_shape)
    dummy_obs_only = np.zeros((1, 1) + obs_shape)

    actor_params = actor_transformed.init(key1, dummy_obs_only[0])
    actor_fwd = actor_transformed.apply

    critic_params = critic_transformed.init(
        key2, dummy_obs_only, dummy_obs, dummy_actions
    )
    critic_fwd = critic_transformed.apply

    params = [actor_params, critic_params]
    fwd = [actor_fwd, critic_fwd]

    return fwd, params


# endregion

# region Losses


def get_loss_fn(config, actor_fwd, critic_fwd):
    def critic_baseline_loss_fn(params, key, batch):
        def clipped_value_loss(new_values, old_values, returns):
            clipped_values = old_values + jnp.clip(
                new_values - old_values, -config["epsilon"], config["epsilon"]
            )
            v1 = jnp.square(returns - new_values)
            v2 = jnp.square(returns - clipped_values)
            return jnp.mean(jnp.fmax(v1, v2))

        observations = batch["observations"]
        actions = batch["actions"]
        returns = batch["returns"]

        # values B, 1
        values = batch["values"]
        new_values = critic_fwd(params, None, observations, None, None)
        value_loss = clipped_value_loss(new_values, values, returns[..., None])

        # baselines N, B, 1
        baselines = batch["baselines"]
        b_obs, b_actions, b_obs_only = make_baseline_inputs(observations, actions)
        # new_baselines = jax.vmap(critic_fwd, in_axes=(None, None, 0, 0, 0))(
        #     params, None, b_obs_only, b_obs, b_actions
        # )
        new_baselines = critic_fwd(params, None, b_obs_only, b_obs, b_actions)
        baseline_loss = clipped_value_loss(
            new_baselines, baselines, returns[..., None, None]
        )

        return config["baseline_coef"] * baseline_loss + config[
            "value_coef"
        ] * value_loss, {"value_loss": value_loss, "baseline_loss": baseline_loss}

    def policy_loss_fn(params, key, batch):
        def clipped_policy_loss(ratios, advantages):
            clipped_ratios = jnp.clip(
                ratios, 1.0 - config["epsilon"], 1.0 + config["epsilon"]
            )
            clipped_objectives = jnp.fmin(
                ratios * advantages, clipped_ratios * advantages
            )
            return -jnp.mean(clipped_objectives)

        observations = batch["observations"]
        log_probs = batch["log_probs"]
        advantages = batch["advantages"]

        # B, N, 1 # WARN : works because only vector observations
        _, new_log_probs = actor_fwd(params, key, observations)
        log_ratios = new_log_probs - log_probs
        ratios = jnp.exp(log_ratios)

        policy_loss = clipped_policy_loss(ratios, advantages)

        return policy_loss, {"policy_loss": policy_loss}

    return policy_loss_fn, critic_baseline_loss_fn


def get_prepare_data_fn(config, critic_fwd):
    def prepare_data_fn(buffer_data, critic_params):
        data = buffer_data
        observations = data["observations"]
        values = critic_fwd(critic_params, None, observations, None, None)  # B, 1

        actions = data["actions"]
        b_obs, b_actions, b_obs_only = make_baseline_inputs(observations, actions)
        baselines = critic_fwd(critic_params, None, b_obs_only, b_obs, b_actions)

        data["values"] = values
        data["baselines"] = baselines

        rewards = data["rewards"]
        discounts = config["gamma"] * (1.0 - data["dones"])
        returns = lambda_returns(
            rewards, discounts, values[..., 0], config["lambda"], True
        )
        advantages = (
            jnp.broadcast_to(returns[..., None, None], baselines.shape) - baselines
        )  # N, B, 1

        data["returns"] = returns
        data["advantages"] = advantages

        return data

    return prepare_data_fn


# endregion

# region POCA


class POCA:
    def __init__(self, config: dict) -> None:
        self.key = jrd.PRNGKey(config["seed"])

        fwd, params = get_continuous_networks(config, self._next_rng_key())
        self.actor_fwd, self.critic_fwd = fwd
        self.actor_params, self.critic_params = params

        self.init_optimizer(config["actor_lr"], config["critic_lr"])

        self.actor_loss_fn, self.critic_loss_fn = get_loss_fn(
            config, self.actor_fwd, self.critic_fwd
        )
        self.prepare_data_fn = get_prepare_data_fn(config, self.critic_fwd)

        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]

    def get_action(self, observations):
        return self.actor_fwd(self.actor_params, self._next_rng_key(), observations)

    def get_value(self, observations):
        return self.critic_fwd(
            self.critic_params,
            self._next_rng_key(),
            np.expand_dims(observations, axis=0),
        )[0]

    def get_baseline(self, observations, actions):
        observations = np.expand_dims(observations, axis=0)
        actions = np.expand_dims(actions, axis=0)
        b_obs, b_actions, b_obs_only = make_baseline_inputs(observations, actions)
        return self.critic_fwd(
            self.critic_params, self._next_rng_key(), b_obs_only, b_obs, b_actions
        )

    def init_optimizer(self, actor_lr, critic_lr):
        self.actor_optimizer = optax.adam(actor_lr)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        self.critic_optimizer = optax.adam(critic_lr)
        self.critic_opt_state = self.critic_optimizer.init(self.critic_params)

    def improve(self, buffer):
        data = self.prepare_data_fn(buffer.sample(), self.critic_params)
        n_batchs = max(len(data) // self.batch_size, 1)

        actor_losses = []
        critic_losses = []
        idx = np.arange(len(data))
        for e in range(self.n_epochs):
            idx = jrd.permutation(self._next_rng_key(), idx, independent=True)
            for i in range(n_batchs):
                batch = get_batch(data, idx)
                (
                    self.actor_params,
                    self.actor_opt_state,
                    (actor_loss, actor_loss_dict),
                ) = update(
                    self.actor_params,
                    self._next_rng_key(),
                    batch,
                    self.actor_loss_fn,
                    self.actor_optimizer,
                    self.actor_opt_state,
                )
                (
                    self.critic_params,
                    self.critic_opt_state,
                    (critic_loss, critic_loss_dict),
                ) = update(
                    self.critic_params,
                    self._next_rng_key(),
                    batch,
                    self.critic_loss_fn,
                    self.critic_optimizer,
                    self.critic_opt_state,
                )

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        logs = {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "critic_loss": sum(critic_losses) / len(critic_losses),
        }
        return logs

    def _next_rng_key(self):
        self.key, _key = jrd.split(self.key)
        return _key


# endregion


# region utils
def make_baseline_inputs(observations, actions):
    new_obs = []
    new_actions = []
    new_obs_only = []

    n_agents = actions.shape[1]
    for i in range(n_agents):
        inds = list(range(n_agents))
        inds.pop(i)
        i_excluded = np.array(inds, np.int32)

        new_obs.append(observations[:, i_excluded])
        new_actions.append(actions[:, i_excluded])
        new_obs_only.append(observations[:, i : i + 1])
    # N * [B, N-1 / 1, size_O / size_A]
    new_obs = jnp.stack(new_obs, axis=1)
    new_actions = jnp.stack(new_actions, axis=1)
    new_obs_only = jnp.stack(new_obs_only, axis=1)
    # B, N, size_O / size_A
    return new_obs, new_actions, new_obs_only


def get_batch(data: dict, inds):
    batch = {}
    for key, value in data.items():
        batch[key] = value[inds]
    return batch


# endregion

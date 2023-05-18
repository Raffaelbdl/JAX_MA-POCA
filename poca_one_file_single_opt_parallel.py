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
from rlax import lambda_returns, clipped_surrogate_pg_loss

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


class ActorTanh(hk.Module):
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


class ActorVanil(hk.Module):
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
        log_stds = hk.get_parameter(
            "log_stds", shape=(1, self.n_actions), dtype=jnp.float32, init=jnp.zeros
        )
        log_stds = jnp.broadcast_to(log_stds, locs.shape)

        return self.get_action(locs, log_stds)

    def get_action(self, locs, log_stds):
        stds = jnp.exp(log_stds)
        normal_dist = dx.Normal(locs, stds)
        samples, log_probs = normal_dist.sample_and_log_prob(seed=hk.next_rng_key())

        return samples, log_probs, normal_dist.entropy(), normal_dist


class Baseline(hk.Module):
    """Baseline module

    Inputs:
        a batch of observations_only (B, 1, size)
        a batch of observations (B, N - 1, size)
        a batch of actions (B, N - 1, size)
    """

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        self.obs_encoder = LinearEncoder(config, "baseline_obs_encoder")
        self.obs_actions_encoder = LinearEncoder(config, "baseline_action_obs_encoder")
        self.rsa = ResidualSelfAttention(config, "baseline_rsa")

    def __call__(self, obs_only: jnp.ndarray, obs: jnp.ndarray, actions: jnp.ndarray):
        rsa_inputs = self.obs_encoder(obs_only)

        obs_actions = jnp.concatenate([obs, actions], axis=-1)
        embed_obs_actions = self.obs_actions_encoder(obs_actions)
        # N_agents axis
        rsa_inputs = jnp.concatenate([rsa_inputs, embed_obs_actions], axis=-2)

        # we assume that there are no nans in obs
        rsa_outputs = self.rsa(rsa_inputs)
        avg_outputs = jnp.mean(rsa_outputs, axis=(-2))

        # value head
        x = jnn.relu(hk.Linear(256)(avg_outputs))
        x = jnn.relu(hk.Linear(256)(x))

        return hk.Linear(1)(x)


class Value(hk.Module):
    """Value module

    Inputs:
        a batch of observations (B, N, size)
    """

    def __init__(self, config: dict, name: str | None = None):
        super().__init__(name)

        self.obs_encoder = LinearEncoder(config, "value_obs_encoder")
        self.rsa = ResidualSelfAttention(config, "value_rsa")

    def __call__(self, obs: jnp.ndarray):
        rsa_inputs = self.obs_encoder(obs)

        # we assume that there are no nans in obs
        rsa_outputs = self.rsa(rsa_inputs)
        avg_outputs = jnp.mean(rsa_outputs, axis=(-2))

        # value head
        x = jnn.relu(hk.Linear(256)(avg_outputs))
        x = jnn.relu(hk.Linear(256)(x))

        return hk.Linear(1)(x)


def get_continuous_networks(config: dict, key: jrd.PRNGKeyArray):
    @hk.transform
    def actor_transformed(observations):
        return ActorVanil(config, "actor")(observations)

    @hk.transform
    def baseline_transformed(observations, actions, obs_only):
        return Baseline(config, "baseline")(obs_only, observations, actions)

    @hk.transform
    def value_transformed(observations):
        return Value(config, "value")(observations)

    key1, key2, key3 = jrd.split(key, 3)
    n_agents = config["n_agents"]
    obs_shape = config["observation_space"].shape
    action_shape = (config["n_actions"],)  # TODO

    dummy_obs = np.zeros((1, n_agents - 1) + obs_shape)
    dummy_obs_only = np.zeros((1, 1) + obs_shape)

    actor_params = actor_transformed.init(key1, dummy_obs_only[0])
    actor_fwd = actor_transformed.apply

    d_obs, d_actions, d_obs_only = make_baseline_inputs(
        jnp.zeros((1, n_agents) + obs_shape), jnp.zeros((1, n_agents) + action_shape)
    )
    baseline_params = baseline_transformed.init(key2, d_obs_only, d_obs, d_actions)
    baseline_fwd = baseline_transformed.apply

    value_params = value_transformed.init(
        key3, jnp.concatenate([dummy_obs_only, dummy_obs], axis=1)
    )
    value_fwd = value_transformed.apply

    params = [actor_params, baseline_params, value_params]
    fwd = [actor_fwd, baseline_fwd, value_fwd]

    return fwd, params


# endregion

# region Losses


def get_loss_fn(config, actor_fwd, baseline_fwd, value_fwd):
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
        new_values = value_fwd(params, None, observations)
        value_loss = clipped_value_loss(new_values, values, returns[..., None])

        # baselines B, N, 1
        baselines = batch["baselines"]
        b_obs, b_actions, b_obs_only = make_baseline_inputs(observations, actions)
        new_baselines = baseline_fwd(params, None, b_obs_only, b_obs, b_actions)
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
        actions = batch["actions"]
        log_probs = batch["log_probs"]
        advantages = batch["advantages"]

        _, _, entropy, dist = actor_fwd(params, key, observations)
        new_log_probs = jnp.sum(dist.log_prob(actions), axis=-1, keepdims=True)
        log_ratios = new_log_probs - log_probs
        ratios = jnp.exp(log_ratios)

        policy_loss = clipped_policy_loss(ratios, advantages)
        entropy = jnp.mean(jnp.sum(entropy, axis=-1))

        return policy_loss - config["entropy_coef"] * entropy, {
            "actor_loss": policy_loss,
            "entropy": entropy,
        }

    def loss_fn(params, key, batch):
        critic_baseline_loss, critic_baseline_loss_dict = critic_baseline_loss_fn(
            params, key, batch
        )
        actor_loss, actor_loss_dict = policy_loss_fn(params, key, batch)

        loss = critic_baseline_loss + actor_loss
        loss_dict = critic_baseline_loss_dict
        loss_dict.update(actor_loss_dict)

        return loss, loss_dict

    return loss_fn


def get_prepare_data_fn(config, baseline_fwd, value_fwd):
    def prepare_data_fn(buffer_data, params):
        data = buffer_data
        observations = data["observations"]  # T E N S
        values = value_fwd(params, None, observations)  # T E 1

        actions = data["actions"]
        b_obs, b_actions, b_obs_only = jax.vmap(
            make_baseline_inputs, in_axes=1, out_axes=1
        )(observations, actions)

        baselines = baseline_fwd(params, None, b_obs_only, b_obs, b_actions)

        data["values"] = values
        data["baselines"] = baselines

        rewards = data["rewards"]
        discounts = config["gamma"] * (1.0 - data["dones"])
        returns = jax.vmap(lambda_returns, in_axes=(1, 1, 1, None, None), out_axes=1)(
            rewards, discounts, values[..., 0], config["lambda"], True
        )  # T E
        advantages = (
            jnp.broadcast_to(returns[..., None, None], baselines.shape) - baselines
        )  # T E N 1
        if config["normalize"]:
            advantages -= jnp.mean(advantages)
            advantages /= jnp.std(advantages) + 1e-8

        data["returns"] = returns
        data["advantages"] = advantages

        data["observations"] = rearrange(data["observations"], "t e n s -> (t e) n s")
        data["actions"] = rearrange(data["actions"], "t e n s -> (t e) n s")
        data["log_probs"] = rearrange(data["log_probs"], "t e n s -> (t e) n s")
        data["rewards"] = rearrange(data["rewards"], "t e -> (t e)")
        data["dones"] = rearrange(data["dones"], "t e -> (t e)")
        data["next_observations"] = rearrange(
            data["next_observations"], "t e n s -> (t e) n s"
        )
        data["baselines"] = rearrange(data["baselines"], "t e n s -> (t e) n s")
        data["values"] = rearrange(data["values"], "t e s -> (t e) s")
        data["returns"] = rearrange(data["returns"], "t e -> (t e)")
        data["advantages"] = rearrange(data["advantages"], "t e n s -> (t e) n s")

        return data

    return prepare_data_fn


# endregion

# region POCA


class POCA:
    def __init__(self, config: dict) -> None:
        self.key = jrd.PRNGKey(config["seed"])

        fwd, params = get_continuous_networks(config, self._next_rng_key())
        self.actor_fwd, self.baseline_fwd, self.value_fwd = fwd
        self.params = hk.data_structures.merge(*params)

        print(self.params.keys())

        self.init_optimizer(config["actor_lr"])

        self.loss_fn = get_loss_fn(
            config, self.actor_fwd, self.baseline_fwd, self.value_fwd
        )
        self.prepare_data_fn = get_prepare_data_fn(
            config, self.baseline_fwd, self.value_fwd
        )

        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.n_steps_trajectory = config["n_steps_trajectory"]
        self.n_envs = config["n_envs"]

    def get_action(self, observations):
        return self.actor_fwd(self.params, self._next_rng_key(), observations)

    def get_value(self, observations):
        return self.value_fwd(
            self.params,
            self._next_rng_key(),
            np.expand_dims(observations, axis=0),
        )[0]

    def get_baseline(self, observations, actions):
        observations = np.expand_dims(observations, axis=0)
        actions = np.expand_dims(actions, axis=0)
        b_obs, b_actions, b_obs_only = make_baseline_inputs(observations, actions)
        return self.baseline_fwd(
            self.params, self._next_rng_key(), b_obs_only, b_obs, b_actions
        )

    def init_optimizer(self, actor_lr):
        self.optimizer = optax.adam(actor_lr)
        self.opt_state = self.optimizer.init(self.params)

    def improve(self, buffer):
        data = self.prepare_data_fn(buffer.sample(), self.params)
        n_batchs = self.n_steps_trajectory * self.n_envs // self.batch_size

        actor_losses = []
        value_losses = []
        baseline_losses = []
        entropies = []
        idx = np.arange(len(data))
        for e in range(self.n_epochs):
            idx = jrd.permutation(self._next_rng_key(), idx, independent=True)
            for i in range(n_batchs):
                batch = get_batch(data, idx)
                (
                    self.params,
                    self.opt_state,
                    (loss, loss_dict),
                ) = update(
                    self.params,
                    self._next_rng_key(),
                    batch,
                    self.loss_fn,
                    self.optimizer,
                    self.opt_state,
                )

                actor_losses.append(loss_dict["actor_loss"])
                baseline_losses.append(loss_dict["baseline_loss"])
                value_losses.append(loss_dict["value_loss"])
                entropies.append(loss_dict["entropy"])

        logs = {
            "actor_loss": sum(actor_losses) / len(actor_losses),
            "baseline_loss": sum(baseline_losses) / len(baseline_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies),
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

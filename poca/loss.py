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

from rltools.agent import Agent
from rltools.buffer import get_batch, OnPolicyBuffer
from rltools.update import update, avg
from rlax import lambda_returns

from poca.baseline_input import make_baseline_inputs


def get_loss_fn(config, policy_fwd, baseline_fwd, value_fwd):
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

        new_dist = policy_fwd(params, key, observations)
        new_log_probs = new_dist.log_prob(actions)
        log_ratios = jnp.sum(new_log_probs - log_probs, axis=-1, keepdims=True)
        ratios = jnp.exp(log_ratios)

        policy_loss = clipped_policy_loss(ratios, advantages)
        entropy = jnp.mean(jnp.sum(new_dist.entropy(), axis=-1))
        approx_kl = jax.lax.stop_gradient(jnp.mean((ratios - 1) - log_ratios))

        return policy_loss - config["entropy_coef"] * entropy, {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "ratio": ratios,
        }

    def loss_fn(params, key, batch):
        policy_loss, policy_loss_dict = policy_loss_fn(params, key, batch)
        critic_baseline_loss, critic_baseline_loss_dict = critic_baseline_loss_fn(
            params, key, batch
        )

        loss = critic_baseline_loss + policy_loss
        loss_dict = critic_baseline_loss_dict
        loss_dict.update(policy_loss_dict)

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

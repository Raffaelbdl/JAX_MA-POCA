from typing import List, Optional

import distrax
import haiku as hk
import jax
import jax.nn as nn
import jax.numpy as jnp

import numpy as np


class NatureCNN(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = hk.initializers.Constant(0.0)

    def forward(self, x: jnp.ndarray):
        x = nn.relu(hk.Conv2D(32, 8, 4, w_init=self.w_init, b_init=self.b_init)(x))
        x = nn.relu(hk.Conv2D(64, 4, 2, w_init=self.w_init, b_init=self.b_init)(x))
        x = nn.relu(hk.Conv2D(64, 3, 1, w_init=self.w_init, b_init=self.b_init)(x))
        x = hk.Flatten()(x)
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 5:
            return jax.vmap(self.forward, in_axes=1, out_axes=1)(x)
        if x.ndim == 6:
            return jax.vmap(
                jax.vmap(self.forward, in_axes=1, out_axes=1), in_axes=1, out_axes=1
            )(x)
        return self.forward(x)


class ObservationEncoder(hk.Module):
    def __init__(self, config: dict, visual: bool, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["observation_embed_size"]
        self.visual = visual

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.visual:
            x = NatureCNN("nature_cnn")(x)

        x = nn.relu(hk.Linear(self.embed_size)(x))
        x = nn.relu(hk.Linear(self.embed_size)(x))

        return x


class ObservationActionEncoder(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["observation_embed_size"]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return hk.Linear(self.embed_size)(x)


class ResidualSelfAttention(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["observation_embed_size"]

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


class ValueHead(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.embed_size = config["observation_embed_size"]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x = nn.relu(hk.Linear(256)(x))
        # x = nn.relu(hk.Linear(256)(x))
        return hk.Linear(1)(x)


class CriticNetwork(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.rsa = ResidualSelfAttention(config, "rsa")

        # TODO check the exact architecture
        self.n_obs = len(config["obs_shape"])
        visuals = [len(o) == 3 for o in config["obs_shape"]]

        self.obs_encoders = [
            ObservationEncoder(config, visuals[i], f"obs_{i}")
            for i in range(self.n_obs)
        ]

        self.obs_action_encoders = [
            ObservationActionEncoder(config, f"obs_action_{i}")
            for i in range(self.n_obs)
        ]

        self.value_head = ValueHead(config, "value_head")

        self.embed_size = config["observation_embed_size"]

    def __call__(
        self, obs_only: List[jnp.ndarray], obs: List[jnp.ndarray], actions: jnp.ndarray
    ):
        self_attn_inputs = jnp.concatenate(
            [self.obs_encoders[i](o) for i, o in enumerate(obs_only)],
            axis=-1,
        )

        if obs is not None:
            encoded_obs = [self.obs_encoders[i](o) for i, o in enumerate(obs)]
            encoded_obs_action = [
                self.obs_action_encoders[i](jnp.concatenate([o, actions], axis=-1))
                for i, o in enumerate(encoded_obs)
            ]

            concatenated_obs_action = jnp.concatenate(encoded_obs_action, axis=-1)

            self_attn_inputs = jnp.concatenate(
                [self_attn_inputs, concatenated_obs_action], axis=1
            )

        self_attn_inputs = hk.Linear(self.embed_size)(self_attn_inputs)
        self_attn_mask = jnp.ones(self_attn_inputs.shape)  # assume that no nans in obs

        encoding = self.rsa(self_attn_inputs, self_attn_mask)
        return self.value_head(encoding)


class PolicyNetwork(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.n_obs = len(config["obs_shape"])
        visuals = [len(o) == 3 for o in config["obs_shape"]]

        self.obs_encoders = [
            ObservationEncoder(config, visuals[i], f"obs_{i}")
            for i in range(self.n_obs)
        ]

    def __call__(self, obs: List[jnp.ndarray]):
        states = []
        for i in range(len(obs)):
            states.append(self.obs_encoders[i](obs[i]))

        x = jnp.concatenate(states, axis=-1)
        x = hk.Linear(256)(x)
        x = hk.Linear(256)(x)

        logits = hk.Linear(self.n)(x)

        return distrax.Categorical(logits=logits)

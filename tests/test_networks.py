import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd

from poca.networks import ResidualSelfAttention
from poca.networks import ObservationEncoder
from poca.networks import ValueHead
from poca.networks import CriticNetwork
from poca.networks import PolicyNetwork

from tests import CONFIG


def test_vector_observation_encoder():
    @hk.transform
    def fwd_fn(obs):
        return ObservationEncoder(CONFIG, False)(obs)

    inputs = jnp.zeros((1,) + CONFIG["obs_shape"][0])
    multi_inputs = jnp.zeros((1, CONFIG["n_agents"]) + CONFIG["obs_shape"][0])

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, inputs)
    outputs = fwd_fn.apply(params, None, inputs)
    multi_outputs = fwd_fn.apply(params, None, multi_inputs)

    assert outputs.shape == (1, CONFIG["observation_embed_size"])
    assert multi_outputs.shape == (
        1,
        CONFIG["n_agents"],
        CONFIG["observation_embed_size"],
    )


def test_visual_observation_encoder():
    @hk.transform
    def fwd_fn(obs):
        return ObservationEncoder(CONFIG, True)(obs)

    inputs = jnp.zeros((1,) + CONFIG["obs_shape"][1])
    multi_inputs = jnp.zeros((1, CONFIG["n_agents"]) + CONFIG["obs_shape"][1])

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, inputs)
    outputs = fwd_fn.apply(params, None, inputs)
    multi_outputs = fwd_fn.apply(params, None, multi_inputs)

    assert outputs.shape == (1, CONFIG["observation_embed_size"])
    assert multi_outputs.shape == (
        1,
        CONFIG["n_agents"],
        CONFIG["observation_embed_size"],
    )


def test_residual_self_attention():
    @hk.transform
    def fwd_fn(inputs, mask):
        return ResidualSelfAttention(CONFIG)(inputs, mask)

    inputs = jnp.zeros((1, CONFIG["n_agents"], CONFIG["observation_embed_size"]))
    # see trainers.torch_entities.networks l328 for how to make mask
    mask = jnp.ones((1, CONFIG["n_agents"], 1))

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, inputs, mask)
    outputs = fwd_fn.apply(params, None, inputs, mask)

    assert outputs.shape == (1, 128)


def test_value_head():
    @hk.transform
    def fwd_fn(embed):
        return ValueHead(CONFIG)(embed)

    inputs = jnp.zeros((1, CONFIG["observation_embed_size"]))
    multi_inputs = jnp.zeros((1, CONFIG["n_agents"], CONFIG["observation_embed_size"]))

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, inputs)
    outputs = fwd_fn.apply(params, None, inputs)
    multi_outputs = fwd_fn.apply(params, None, multi_inputs)

    assert outputs.shape == (1, 1)
    assert multi_outputs.shape == (1, CONFIG["n_agents"], 1)


def test_critic_network():
    @hk.transform
    def fwd_fn(obs_only, obs, actions):
        return CriticNetwork(CONFIG)(obs_only, obs, actions)

    obs_only = [jnp.zeros((1, 1) + shape) for shape in CONFIG["obs_shape"]]
    obs = [
        jnp.zeros((1, CONFIG["n_agents"] - 1) + shape) for shape in CONFIG["obs_shape"]
    ]
    actions = jnp.zeros((1, CONFIG["n_agents"] - 1, 1))

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, obs_only, obs, actions)
    outputs = fwd_fn.apply(params, None, obs_only, obs, actions)
    obs_only_outputs = fwd_fn.apply(params, None, obs_only, None, None)

    assert outputs.shape == (1, 1)
    assert obs_only_outputs.shape == (1, 1)

    # used in poca critic baseline loss
    extra_dim_obs_only = [jnp.zeros((1, 3, 2) + shape) for shape in CONFIG["obs_shape"]]
    extra_dim_outputs = jax.vmap(
        fwd_fn.apply, in_axes=(None, None, 1, None, None), out_axes=1
    )(params, None, extra_dim_obs_only, None, None)
    assert extra_dim_outputs.shape == (1, 3, 1)


def test_policy_network():
    @hk.transform
    def fwd_fn(obs):
        return PolicyNetwork(CONFIG)(obs)

    obs = [jnp.zeros((1,) + shape) for shape in CONFIG["obs_shape"]]
    multi_obs = [
        jnp.zeros((1, CONFIG["n_agents"]) + shape) for shape in CONFIG["obs_shape"]
    ]
    T = 4
    time_multi_obs = [
        jnp.zeros((1, T, CONFIG["n_agents"]) + shape) for shape in CONFIG["obs_shape"]
    ]

    key = jrd.PRNGKey(CONFIG["seed"])
    params = fwd_fn.init(key, obs)

    outputs = fwd_fn.apply(params, None, obs)
    multi_outputs = fwd_fn.apply(params, None, multi_obs)
    time_multi_outputs = fwd_fn.apply(params, None, time_multi_obs)

    assert outputs.sample(seed=key).shape == (1,)
    assert multi_outputs.sample(seed=key).shape == (1, CONFIG["n_agents"])
    assert time_multi_outputs.sample(seed=key).shape == (1, T, CONFIG["n_agents"])

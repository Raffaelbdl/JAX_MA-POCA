from typing import Callable, List, Tuple

from einops import rearrange
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax
from rlax._src import multistep

from poca.buffer import POCABuffer
from poca.networks import PolicyNetwork
from poca.networks import CriticNetwork


def get_policy(config) -> Tuple[hk.Transformed, hk.Params]:
    @hk.transform
    def fwd_fn(obs):
        return PolicyNetwork(config, "policy")(obs)

    obs = []
    for shape in config["obs_shape"]:
        obs.append(jnp.zeros((1,) + shape))

    seed = config["seed"]
    key = jrd.PRNGKey(seed)

    params = fwd_fn.init(key, obs)

    return fwd_fn.apply, params


def get_critic(config) -> Tuple[hk.Transformed, hk.Params]:
    @hk.transform
    def fwd_fn(obs_only, obs, actions):
        return CriticNetwork(config, "critic")(obs_only, obs, actions)

    n_agents = config["n_agents"]
    obs_only, obs = [], []
    for shape in config["obs_shape"]:
        obs_only.append(jnp.zeros((1, 1) + shape))
        obs.append(jnp.zeros((1, n_agents - 1) + shape))

    actions = jnp.zeros((1, n_agents - 1, 1))

    seed = config["seed"]
    key = jrd.PRNGKey(seed)

    params = fwd_fn.init(key, obs_only, obs, actions)

    return fwd_fn.apply, params


def get_poca_loss_fn(config, policy_fwd, critic_fwd) -> Callable:
    def critic_baseline_loss(params, batch):
        obs = batch["obs"]  # [B, T, N, size]
        actions = batch["actions"]  # B, T, N
        reward = batch["rewards"]  # B, T
        discount = batch["discounts"]  # B, T
        next_obs = batch["next_obs"]  # [B, T, N, size]

        def get_lambda_returns(r_t, discount_t, o_t):
            v_t = critic_fwd(params, None, o_t, None, None)[..., 0]  # value
            return multistep.lambda_returns(
                r_t, discount_t, v_t, config["lambda"], True
            )

        lambda_returns = jax.vmap(get_lambda_returns, in_axes=0, out_axes=0)(
            reward, discount, next_obs
        )  # B, T

        # values
        values = jax.vmap(
            critic_fwd,
            in_axes=(None, None, 1, None, None),
            out_axes=1,
        )(
            params, None, obs, None, None
        )[..., 0]
        v_loss = -jnp.square(values - lambda_returns)  # B, T

        # baselines
        b_obs_only, b_obs, b_actions = get_baselines_input(
            obs, actions
        )  # B, T, N, size

        baselines = jax.vmap(critic_fwd, in_axes=(None, None, 1, 1, 1), out_axes=1)(
            params, None, b_obs_only, b_obs, b_actions[..., None]
        )[..., 0]

        b_loss = -jnp.square(baselines - lambda_returns)  # B, T

        return config["b_coef"] * jnp.mean(b_loss) + config["v_coef"] * jnp.mean(v_loss)

    def policy_loss(params, key, batch):
        obs = batch["obs"]  # B, T, N, size
        log_probs = batch["log_probs"]  # B, T, N
        advantages = batch["advantages"]  # B, T, N

        dist = policy_fwd(params, None, obs)
        _, new_log_probs = dist.sample_and_log_prob(seed=key)  # B, T, N

        ratio = jnp.exp(new_log_probs - log_probs)  # B, T, N
        return -jnp.mean(ratio * advantages) - config["entropy_coef"] * dist.entropy()

    def poca_loss(params, key, batch):
        return critic_baseline_loss(params, batch) + policy_loss(params, key, batch)

    # TODO JIT
    return poca_loss


class POCA:
    def __init__(self, config: dict):
        self.key = jrd.PRNGKey(config["seed"])

        self.policy_fwd, self.policy_params = get_policy(config)
        self.critic_fwd, self.critic_params = get_critic(config)
        self.params = hk.data_structures.merge(self.policy_params, self.critic_params)

        self.loss_fn = get_poca_loss_fn(config, self.policy_fwd, self.critic_fwd)

        self.buffer = POCABuffer(config)

        self.discount = config["discount"]
        self._lambda = config["lambda"]

    def get_action(self, obs: List[jnp.ndarray]):
        # TODO JIT
        dist = self.policy_fwd(self.params, None, obs)
        return dist.sample_and_log_prob(seed=self._next_rng_key())

    def improve(self):
        pass

    def prepare_data(self):
        buffer = self.buffer.sample()
        data = {}

        data["obs"] = [jnp.stack(obs, axis=0) for obs in buffer["obs"]]  # T, N, size
        data["actions"] = jnp.stack(buffer["actions"], axis=0)
        data["rewards"] = jnp.stack(buffer["rewards"], axis=0)  # T
        data["dones"] = jnp.stack(buffer["dones"], axis=0)  # T
        data["next_obs"] = [jnp.stack(obs, axis=0) for obs in buffer["next_obs"]]
        data["log_probs"] = jnp.stack(buffer["log_probs"], axis=0)

        data["discounts"] = self.discount * jnp.where(data["dones"], 0.0, 1.0)

        next_values = self.critic_fwd(self.params, None, data["next_obs"], None, None)[
            ..., 0
        ]
        lambda_returns = multistep.lambda_returns(
            data["rewards"], data["discounts"], next_values, self._lambda, True
        )

        b_obs_only, b_obs, b_actions = get_baselines_input(
            [obs[None, ...] for obs in data["obs"]], data["actions"][None, ...]
        )  # 1, T, N, size

        baselines = jax.vmap(
            self.critic_fwd, in_axes=(None, None, 1, 1, 1), out_axes=1
        )(self.params, None, b_obs_only, b_obs, b_actions[..., None])[0, ..., 0]

        data["advantages"] = lambda_returns - baselines

        return data

    def _next_rng_key(self):
        self.key, key1 = jrd.split(self.key)
        return key1


def get_baselines_input(
    obs: List[jnp.ndarray], actions
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # actions shape : B, T, N
    # obs shape : [B, T, N, size]
    n_agents = actions.shape[2]

    new_obs_only = []
    new_obs = []
    new_actions = []

    for i in range(n_agents):
        _new_obs_only = []
        _new_obs = []
        i_excluded = list(range(n_agents)).pop(i)

        for o in obs:
            _new_obs_only.append(o[:, :, i])
            _new_obs.append(o[:, :, i_excluded])

        new_obs_only.append(_new_obs_only)
        new_obs.append(_new_obs)
        new_actions.append(actions[:, :, i_excluded])

    # B, T, N, size
    new_obs_only = zip(*new_obs_only)
    new_obs_only = [jnp.stack(noo, axis=2) for noo in new_obs_only]

    new_obs = zip(*new_obs)
    new_obs = [jnp.stack(no, axis=2) for no in new_obs]

    new_actions = jnp.stack(new_actions, axis=2)

    return new_obs_only, new_obs, new_actions

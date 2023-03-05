from typing import Callable, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import numpy as np
from rlax import clipped_surrogate_pg_loss
from rlax._src import multistep

from poca.buffer import SimpleAgentBuffer
from poca.networks import PolicyNetwork
from poca.networks import CriticNetwork

from minimalistic_rl.utils import apply_updates


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
        # trust region value loss
        def clipped_loss(new_values, old_values, returns):
            clipped_values = old_values + jnp.clip(
                new_values - old_values, -config["epsilon"], config["epsilon"]
            )
            v1 = jnp.square(returns - values)
            v2 = jnp.square(returns - clipped_values)
            return jnp.mean(jnp.fmax(v1, v2))

        obs = batch["obs"]  # [B, N, size]
        actions = batch["actions"]  # B, N
        returns = batch["returns"]

        # values
        old_values = batch["values"]
        values = critic_fwd(params, None, obs, None, None)[..., 0]

        v_loss = clipped_loss(values, old_values, returns)

        # baselines
        old_baselines = batch["baselines"]
        b_obs_only, b_obs, b_actions = get_baselines_input(obs, actions)  # [B, N, size]

        baselines = critic_fwd(params, None, b_obs_only, b_obs, b_actions[..., None])[
            ..., 0
        ]

        b_loss = clipped_loss(baselines, old_baselines, returns)

        return config["b_coef"] * b_loss + config["v_coef"] * v_loss

    def policy_loss(params, key, batch):
        # trust region policy loss

        obs = batch["obs"]  # B, N, size
        log_probs = batch["log_probs"]  # B, N
        advantages = batch["advantages"]  # B,

        dist = policy_fwd(params, None, obs)
        _, new_log_probs = dist.sample_and_log_prob(seed=key)  # B, N

        ratio = jnp.exp(new_log_probs - log_probs)  # B, N

        policy_loss = jax.vmap(
            clipped_surrogate_pg_loss, in_axes=(1, None, None, None)
        )(ratio, advantages, config["epsilon"], True)

        return jnp.mean(policy_loss) - config["entropy_coef"] * jnp.mean(dist.entropy())

    def poca_loss(params, key, batch):
        return critic_baseline_loss(params, batch) + policy_loss(params, key, batch)

    return poca_loss


class POCA:
    def __init__(self, config: dict):
        self.key = jrd.PRNGKey(config["seed"])

        self.policy_fwd, self.policy_params = get_policy(config)
        self.critic_fwd, self.critic_params = get_critic(config)
        self.params = hk.data_structures.merge(self.policy_params, self.critic_params)

        self.loss_fn = get_poca_loss_fn(config, self.policy_fwd, self.critic_fwd)
        self.loss_fn = jax.jit(self.loss_fn)

        self.buffer = SimpleAgentBuffer()

        self.discount = config["discount"]
        self._lambda = config["lambda"]

        self.buffer_capacity = config["buffer_capacity"]
        self.n_epochs = config["n_epochs"]
        self.n_minibatchs = config["n_minibatchs"]
        self.batch_size = self.buffer_capacity // self.n_minibatchs

        self.learning_rate = config["learning_rate"]
        self.optimizer, self.opt_state = self.init_optimizer()

    def get_action(self, obs: List[jnp.ndarray]):
        dist = jax.jit(self.policy_fwd)(self.params, None, obs)
        return dist.sample_and_log_prob(seed=self._next_rng_key())

    def improve(self, logs: dict):
        data = self.prepare_data()
        idx = np.arange(len(data["rewards"]))

        loss_list = []
        for e in range(self.n_epochs):
            idx = jrd.permutation(self._next_rng_key(), idx, independent=True)
            for i in range(self.n_minibatchs):
                _idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
                batch = get_batch(data, _idx)

                loss, grads = jax.value_and_grad(self.loss_fn)(
                    self.params, self._next_rng_key(), batch
                )

                self.params, self.opt_state = apply_updates(
                    self.optimizer, self.params, self.opt_state, grads
                )

                loss_list.append(loss)

        logs["loss"] = sum(loss_list) / len(loss_list)

        self.buffer.reset()

        return logs

    @property
    def improve_condition(self) -> bool:
        return len(self.buffer) >= self.buffer_capacity

    def prepare_data(self):
        buffer = self.buffer.sample()
        data = {}
        # T * [types * [N, size']] --> types * [T, N, size']
        obs = [[] for _ in range(len(buffer["obs"][0]))]
        next_obs = [[] for _ in range(len(buffer["obs"][0]))]
        for t in range(len(buffer["obs"])):
            o = buffer["obs"][t]
            no = buffer["next_obs"][t]
            for i in range(len(o)):
                obs[i].append(o[i])
                next_obs[i].append(no[i])

        data["obs"] = [jnp.stack(o, axis=0) for o in obs]
        data["next_obs"] = [jnp.stack(o, axis=0) for o in next_obs]

        data["actions"] = jnp.stack(buffer["actions"], axis=0)
        data["rewards"] = jnp.stack(buffer["rewards"], axis=0)  # T
        data["dones"] = jnp.stack(buffer["dones"], axis=0)  # T
        data["log_probs"] = jnp.stack(buffer["log_probs"], axis=0)

        data["discounts"] = self.discount * jnp.where(data["dones"], 0.0, 1.0)
        next_values = jax.jit(self.critic_fwd)(
            self.params, None, data["next_obs"], None, None
        )[..., 0]
        lambda_returns = jax.jit(multistep.lambda_returns, static_argnums=(4))(
            data["rewards"], data["discounts"], next_values, self._lambda, True
        )

        b_obs_only, b_obs, b_actions = jax.jit(get_baselines_input)(
            [obs for obs in data["obs"]], data["actions"]
        )  # [T, N, size]

        baselines = jax.jit(self.critic_fwd)(
            self.params, None, b_obs_only, b_obs, b_actions[..., None]
        )[..., 0]

        values = jax.jit(self.critic_fwd)(self.params, None, data["obs"], None, None)[
            ..., 0
        ]

        data["returns"] = lambda_returns
        data["advantages"] = lambda_returns - baselines

        data["baselines"] = baselines
        data["values"] = values

        return data

    def init_optimizer(self):
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)
        return optimizer, opt_state

    def _next_rng_key(self):
        self.key, key1 = jrd.split(self.key)
        return key1


def get_baselines_input(
    obs: List[jnp.ndarray], actions
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # actions shape : B, N
    # obs shape : [B, N, size]
    n_agents = actions.shape[1]

    new_obs_only = []
    new_obs = []
    new_actions = []

    for i in range(n_agents):
        _new_obs_only = []
        _new_obs = []
        i_excluded = list(range(n_agents)).pop(i)

        for o in obs:
            _new_obs_only.append(o[:, i])
            _new_obs.append(o[:, i_excluded])

        new_obs_only.append(_new_obs_only)
        new_obs.append(_new_obs)
        new_actions.append(actions[:, i_excluded])

    # B, N, size
    new_obs_only = zip(*new_obs_only)
    new_obs_only = [jnp.stack(noo, axis=1) for noo in new_obs_only]

    new_obs = zip(*new_obs)
    new_obs = [jnp.stack(no, axis=1) for no in new_obs]

    new_actions = jnp.stack(new_actions, axis=1)

    return new_obs_only, new_obs, new_actions


def get_batch(data: dict, idx: np.array):
    batch = {}
    for key, value in data.items():
        if key in ("obs", "next_obs"):
            batch[key] = [v[idx] for v in value]
        else:
            batch[key] = value[idx]

    return batch

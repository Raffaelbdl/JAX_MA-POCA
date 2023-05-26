"""POCA Agent"""
import haiku as hk
import jax
import jax.random as jrd
import numpy as np
import optax

from rltools.agent import Agent
from rltools.buffer import get_batch, OnPolicyBuffer
from rltools.types import Array
from rltools.update import update, avg

from poca.networks import get_networks_fn
from poca.loss import get_loss_fn, get_prepare_data_fn


class POCA(Agent):
    def __init__(self, config: dict) -> None:
        super().__init__(jrd.PRNGKey(config["seed"]))

        fwd, params = get_networks_fn(config)
        self.policy_fwd, self.baseline_fwd, self.value_fwd = fwd
        self.params = hk.data_structures.merge(*params)

        self.loss_fn = jax.jit(
            get_loss_fn(config, self.policy_fwd, self.baseline_fwd, self.value_fwd)
        )
        self.prepare_data_fn = jax.jit(
            get_prepare_data_fn(config, self.baseline_fwd, self.value_fwd)
        )

        self.n_epochs = config["n_epochs"]
        self.batch_size = config["batch_size"]
        self.n_batchs = config["n_envs"] * config["n_steps_per_trajectory"]
        self.n_batchs //= self.batch_size

        self.init_optimizer(config)

    def get_action(self, observations) -> tuple[Array, Array]:
        dists = jax.jit(self.policy_fwd)(
            self.params, self._next_rng_key(), observations
        )
        return dists.sample_and_log_prob(seed=self._next_rng_key())

    def improve(self, buffer: OnPolicyBuffer):
        policy_loss = []
        entropy = []
        approx_kl = []
        value_loss = []
        baseline_loss = []

        data = self.prepare_data_fn(buffer.sample(), self.params)
        inds = np.arange(len(data["rewards"]))

        for e in range(self.n_epochs):
            inds = jrd.permutation(self._next_rng_key(), inds, independent=True)

            for i in range(self.n_batchs):
                _inds = inds[i * self.batch_size : (i + 1) * self.batch_size]
                batch = get_batch(data, _inds)

                (self.params, self.opt_state, (loss, loss_dict)) = update(
                    self.params,
                    self._next_rng_key(),
                    batch,
                    self.loss_fn,
                    self.optimizer,
                    self.opt_state,
                )

                policy_loss.append(loss_dict["policy_loss"])
                entropy.append(loss_dict["entropy"])
                approx_kl.append(loss_dict["approx_kl"])
                value_loss.append(loss_dict["value_loss"])
                baseline_loss.append(loss_dict["baseline_loss"])

        return {
            "policy_loss": avg(policy_loss),
            "approx_kl": avg(approx_kl),
            "entropy": avg(entropy),
            "value_loss": avg(value_loss),
            "baseline": avg(baseline_loss),
        }

    def init_optimizer(self, config):
        self.optimizer = optax.adam(config["learning_rate"])
        self.opt_state = self.optimizer.init(self.params)


def collect_trajectory(envs, agent: POCA, n_steps_trajectory, last_observations, logs):
    """env is an envpool environment"""
    buffer = OnPolicyBuffer()
    observations = last_observations

    for t in range(n_steps_trajectory):
        actions, log_probs = agent.get_action(observations)
        next_observations, rewards, dones, truncs, info = envs.step(actions)

        logs["tmp_episode_reward"] += rewards

        buffer.add(observations, actions, log_probs, rewards, dones, next_observations)
        observations = next_observations

        for i, (d, t) in enumerate(zip(dones, truncs)):
            if d or t:
                logs["avg_episode_reward"].append(logs["tmp_episode_reward"][i])
                logs["tmp_episode_reward"][i] = 0.0

    return buffer, next_observations, logs

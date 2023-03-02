import jax.numpy as jnp
import jax.random as jrd

from poca.poca import POCA

from tests import CONFIG


def test_poca_loss():
    poca = POCA(CONFIG)

    B = 5
    T = 4
    N = CONFIG["n_agents"]

    batch = {
        "obs": [jnp.zeros((B, T, N) + shape) for shape in CONFIG["obs_shape"]],
        "actions": jnp.zeros((B, T, N)),
        "rewards": jnp.zeros((B, T)),
        "discounts": jnp.zeros((B, T)),
        "next_obs": [jnp.zeros((B, T, N) + shape) for shape in CONFIG["obs_shape"]],
        "log_probs": jnp.zeros((B, T, N)),
        "advantages": jnp.zeros((B, T, N)),
    }
    key = jrd.PRNGKey(CONFIG["seed"])
    loss = poca.loss_fn(poca.params, key, batch)

    assert True


def test_poca_get_actions():
    poca = POCA(CONFIG)

    obs = [jnp.zeros((CONFIG["n_agents"],) + shape) for shape in CONFIG["obs_shape"]]
    actions, log_probs = poca.get_action(obs)

    assert actions.shape == (CONFIG["n_agents"],)
    assert log_probs.shape == (CONFIG["n_agents"],)


def test_poca_prepare_data():
    poca = POCA(CONFIG)

    T = 4
    N = CONFIG["n_agents"]
    poca.buffer.buffer = {
        "obs": [
            [jnp.zeros((N,) + shape) for _ in range(T)] for shape in CONFIG["obs_shape"]
        ],
        "actions": [jnp.zeros((N)) for _ in range(T)],
        "rewards": [jnp.zeros(()) for _ in range(T)],
        "dones": [jnp.zeros(()) for _ in range(T)],
        "next_obs": [
            [jnp.zeros((N,) + shape) for _ in range(T)] for shape in CONFIG["obs_shape"]
        ],
        "log_probs": [jnp.zeros((N)) for _ in range(T)],
    }

    data = poca.prepare_data()
    for i, shape in enumerate(CONFIG["obs_shape"]):
        assert data["obs"][i].shape == (T, N) + shape
        assert data["next_obs"][i].shape == (T, N) + shape
    assert data["actions"].shape == (T, N)
    assert data["rewards"].shape == (T,)
    assert data["discounts"].shape == (T,)
    assert data["log_probs"].shape == (T, N)
    assert data["advantages"].shape == (T,)

SEED = 0
OBSERVATION_EMBED_SIZE = 128
N_AGENTS = 3
N_ACTIONS = 5

CONFIG = {
    "seed": SEED,
    "obs_shape": [(21,), (20, 20, 6)],
    "observation_embed_size": OBSERVATION_EMBED_SIZE,
    "n_agents": N_AGENTS,
    "n_actions": N_ACTIONS,
    "lambda": 0.95,
    "discount": 0.99,
    "b_coef": 1.0,
    "v_coef": 0.5,
    "entropy_coef": 0.02,
}

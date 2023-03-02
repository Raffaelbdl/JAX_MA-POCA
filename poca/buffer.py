class POCABuffer:
    def __init__(self, config: dict) -> None:
        self.reset()

    def reset(self):
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_obs": [],
            "log_probs": [],
        }

    def add(self, obs, actions, rewards, dones, next_obs, log_probs):
        self.buffer["obs"].append(obs)
        self.buffer["actions"].append(actions)
        self.buffer["rewards"].append(rewards)
        self.buffer["dones"].append(dones)
        self.buffer["next_obs"].append(next_obs)
        self.buffer["log_probs"].append(log_probs)

    def sample(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer["rewards"])

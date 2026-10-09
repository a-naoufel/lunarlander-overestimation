from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Config:
    # --- run ---
    algo: str = "td3"                   # "ddpg" | "td3" (Stable-Baselines3 implementations)
    env: str = "LunarLander-v3"         # created with continuous=True, see models.make_env
    seed: int = 0
    total_steps: int = 150_000
    learning_starts: int = 5_000        # uniform random actions before this step
    threads: int = 1                    # torch threads (1 is best when running seeds in parallel)

    # --- optimisation ---
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.98
    tau: float = 0.01
    lr: float = 1e-3                    # SB3 uses a single learning rate for the actor and the critics
    utd: int = 1                        # gradient steps per environment step

    # --- networks ---
    hidden: int = 256
    n_layers: int = 2                   # number of hidden layers (2 or 3)
    critic_ln: bool = False             # LayerNorm between each hidden Linear and its ReLU (critics)
    actor_ln: bool = False              # same for the actor

    # --- algorithm ---
    expl_noise: float = 0.1             # std of the Gaussian exploration noise (both algorithms)
    # SB3's DDPG is TD3 with its three tricks switched off, so these are TD3-only (ignored by DDPG):
    n_critics: Optional[int] = None     # Q networks in the clipped-min target (None: 2 for td3, 1 for ddpg)
    policy_delay: Optional[int] = None  # critic steps per actor step       (None: 2 for td3, 1 for ddpg)
    target_noise: float = 0.2           # target policy smoothing: std ...
    noise_clip: float = 0.5             # ... and clip of the noise

    # --- evaluation and Monte Carlo bias estimation ---
    eval_every: int = 5_000
    eval_episodes: int = 10
    mc_min_horizon: int = 300           # truncated episodes: only use states with >= this many steps left (0.98^300 ~ 0.002)
    final_window: int = 5               # summary.json averages the last k evaluations

    # --- logging ---
    log_every: int = 1_000              # env steps between two logs of the networks' internals
    tensorboard: bool = True
    save_raw: bool = True               # per evaluation: raw (Q, MC return) of every visited state
    save_ckpt: bool = False             # also save the weights at every evaluation (final.pt is always saved)

    def __post_init__(self):
        if self.algo not in ("ddpg", "td3"):
            raise ValueError(f"unknown algo {self.algo!r}")
        td3 = self.algo == "td3"
        if self.n_critics is None:
            self.n_critics = 2 if td3 else 1
        if self.policy_delay is None:
            self.policy_delay = 2 if td3 else 1
        if not td3 and (self.n_critics != 1 or self.policy_delay != 1):
            raise ValueError("SB3's DDPG has one critic and no policy delay; use algo=td3 to vary them")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

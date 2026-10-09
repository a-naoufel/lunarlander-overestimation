import numpy as np


def discounted_returns(rewards, gamma):
    """G_t = r_t + gamma * G_{t+1} for every t of one episode."""
    g, out = 0.0, np.zeros(len(rewards))
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


class Evaluator:
    """Deterministic rollouts of the current policy. Returns

    - performance: undiscounted episode returns,
    - overestimation bias: Q(s, a) minus the discounted Monte Carlo return G(s, a) of the same policy,
      for every (s, a) visited during the rollouts.

    Terminated episodes give exact MC returns. A time-limit truncation leaves no exact return near the
    end of the episode, so only states with at least `mc_min_horizon` steps left are used there.
    """

    def __init__(self, cfg, env):
        self.cfg, self.env = cfg, env
        self.rng = np.random.default_rng(cfg.seed + 10_000)

    def run(self, agent):
        c = self.cfg
        returns, S, A, G, episode, t_idx = [], [], [], [], [], []
        for ep in range(c.eval_episodes):
            s, _ = self.env.reset(seed=int(self.rng.integers(2**31)))
            states, actions, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                a = agent.act(s)
                states.append(s)
                actions.append(a)
                s, r, terminated, truncated, _ = self.env.step(a)
                rewards.append(r)
            returns.append(float(np.sum(rewards)))

            keep = len(rewards) if terminated else max(0, len(rewards) - c.mc_min_horizon)
            S += states[:keep]
            A += actions[:keep]
            G.append(discounted_returns(rewards, c.gamma)[:keep])
            episode += [ep] * keep
            t_idx += range(keep)

        metrics = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
                   "n_mc_states": len(S)}
        raw = {"returns": np.array(returns, dtype=np.float32)}
        if S:
            q = agent.q_values(np.array(S), np.array(A))             # (N, n_heads)
            g = np.concatenate(G)
            q1, qmin = q[:, 0], q.min(1)
            err = q1 - g
            metrics.update(
                q_mean=float(q1.mean()),
                mc_mean=float(g.mean()),
                bias=float(err.mean()),                              # head the actor follows
                bias_std=float(err.std()),
                bias_norm=float(err.mean() / (abs(g.mean()) + 1e-6)),  # relative to the return scale
                bias_min=float((qmin - g).mean()),                   # min over heads (= bias for DDPG)
                abs_err=float(np.abs(err).mean()),
                q_gap=float((q.max(1) - qmin).mean()) if q.shape[1] > 1 else float("nan"),
            )
            raw.update(q=q.astype(np.float32), g=g.astype(np.float32),
                       episode=np.array(episode), t=np.array(t_idx))
        return metrics, raw

"""DDPG / TD3 on LunarLanderContinuous with optional LayerNorm and Monte Carlo
estimation of the critic's overestimation bias.

Single file, CleanRL style. DDPG here is TD3 without its three tricks
("OurDDPG" in Fujimoto et al. 2018): single critic, no target policy
smoothing, actor updated every critic step. Everything else (networks,
exploration noise, optimiser, batch size) is shared so that the comparison
isolates the algorithmic differences.
"""
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ddpg", "td3"], default="td3")
    p.add_argument("--env", default="LunarLanderContinuous-v3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2, help="number of hidden layers")
    p.add_argument("--critic-ln", type=int, default=0, help="LayerNorm in critic hidden layers")
    p.add_argument("--actor-ln", type=int, default=0, help="LayerNorm in actor hidden layers")
    p.add_argument("--utd", type=int, default=1, help="critic gradient steps per env step")
    p.add_argument("--policy-delay", type=int, default=None,
                   help="critic updates per actor update (default: 2 for td3, 1 for ddpg)")
    p.add_argument("--expl-noise", type=float, default=0.1)
    p.add_argument("--target-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--twin", type=int, default=None,
                   help="two critics, target from their min (TD3 change 1); default: 1 for td3, 0 for ddpg")
    p.add_argument("--smooth", type=int, default=None,
                   help="target policy smoothing (TD3 change 2); default: 1 for td3, 0 for ddpg")
    p.add_argument("--beta", type=float, default=1.0,
                   help="TD3 target uses beta*min(Q1',Q2') + (1-beta)*max(Q1',Q2'); 1 = standard TD3")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--mc-min-horizon", type=int, default=500,
                   help="in truncated episodes, only use states with at least this many "
                        "remaining steps for the MC bias (gamma^500 ~ 0.007)")
    p.add_argument("--out", default="runs")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--save-ckpt", type=int, default=0,
                   help="save actor+critic weights at every evaluation (for watch.py)")
    a = p.parse_args()
    if a.policy_delay is None:
        a.policy_delay = 2 if a.algo == "td3" else 1
    for flag in ("twin", "smooth"):  # the defaults reproduce plain DDPG and TD3
        if getattr(a, flag) is None:
            setattr(a, flag, int(a.algo == "td3"))
    if not a.twin and a.beta != 1:
        p.error("--beta needs two critics (td3, or --twin 1)")
    return a


def mlp(inp, out, hidden, n_layers, ln):
    layers, d = [], inp
    for _ in range(n_layers):
        layers.append(nn.Linear(d, hidden))
        if ln:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, a):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, a.hidden, a.n_layers, a.actor_ln)

    def forward(self, s):
        return torch.tanh(self.net(s))  # LunarLander actions are in [-1, 1]


class Critic(nn.Module):
    """One or two Q heads (two for TD3's clipped double Q)."""

    def __init__(self, obs_dim, act_dim, a, n_heads):
        super().__init__()
        self.qs = nn.ModuleList(
            mlp(obs_dim + act_dim, 1, a.hidden, a.n_layers, a.critic_ln) for _ in range(n_heads))

    def forward(self, s, act):
        x = torch.cat([s, act], -1)
        return torch.cat([q(x) for q in self.qs], -1)  # (B, n_heads)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng, bs):
        idx = rng.integers(0, self.n, bs)
        return [torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d)]


@torch.no_grad()
def evaluate(env, actor, critic, a, rng):
    """Deterministic rollouts of the current policy.

    Returns undiscounted episode returns (performance) and, for every visited
    (s, a), the critic's Q(s, a) and the discounted Monte Carlo return G(s, a)
    of the same policy, i.e. the quantity the critic is trying to estimate.
    """
    returns, qs, gs = [], [], []
    for _ in range(a.eval_episodes):
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        S, A, R = [], [], []
        done = trunc = False
        while not (done or trunc):
            act = actor(torch.as_tensor(s, dtype=torch.float32)[None])[0].numpy()
            S.append(s); A.append(act)
            s, r, done, trunc, _ = env.step(act)
            R.append(r)
        returns.append(float(np.sum(R)))
        G = np.zeros(len(R))
        g = 0.0
        for t in reversed(range(len(R))):
            g = R[t] + a.gamma * g
            G[t] = g
        # A truncated episode has no exact MC return near its end: drop those states.
        keep = len(R) if done else max(0, len(R) - a.mc_min_horizon)
        if keep:
            q = critic(torch.as_tensor(np.array(S[:keep]), dtype=torch.float32),
                       torch.as_tensor(np.array(A[:keep]), dtype=torch.float32))
            qs.append(q.numpy()); gs.append(G[:keep])
    out = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
           "returns": returns, "n_mc_states": int(sum(len(g) for g in gs))}
    if gs:
        Q = np.concatenate(qs); G = np.concatenate(gs)
        Q1 = Q[:, 0]
        Qmin = Q.min(1)
        scale = abs(G.mean()) + 1e-6
        out.update({
            "q_mean": float(Q1.mean()), "mc_mean": float(G.mean()),
            "bias": float((Q1 - G).mean()),              # Q1: the head the actor follows
            "bias_std": float((Q1 - G).std()),
            "bias_norm": float((Q1 - G).mean() / scale),  # normalised as in Chen et al. 2021 (REDQ)
            "bias_min": float((Qmin - G).mean()),          # = bias for DDPG; TD3's target estimate
            "abs_err": float(np.abs(Q1 - G).mean()),
            # disagreement of the two critics; bias of the beta target = bias_min + (1-beta)*q_gap
            "q_gap": float((Q.max(1) - Qmin).mean()),
        })
    return out


def main():
    a = parse_args()
    torch.set_num_threads(a.threads)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    eval_rng = np.random.default_rng(a.seed + 10_000)

    default = int(a.algo == "td3")
    name = (f"{a.algo}_ln{a.critic_ln}{a.actor_ln}_L{a.n_layers}_utd{a.utd}_pd{a.policy_delay}"
            + (f"_tw{a.twin}" if a.twin != default else "") + (f"_sm{a.smooth}" if a.smooth != default else "")
            + (f"_b{a.beta:g}" if a.beta != 1 else "") + f"_s{a.seed}")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, name + ".json")
    ckpt_path, ckpts = os.path.join(a.out, name + ".pt"), {}

    env = gym.make(a.env)
    eval_env = gym.make(a.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.action_space.seed(a.seed)

    n_heads = 2 if a.twin else 1
    actor = Actor(obs_dim, act_dim, a)
    critic = Critic(obs_dim, act_dim, a, n_heads)
    actor_t = Actor(obs_dim, act_dim, a); actor_t.load_state_dict(actor.state_dict())
    critic_t = Critic(obs_dim, act_dim, a, n_heads); critic_t.load_state_dict(critic.state_dict())
    for p in (*actor_t.parameters(), *critic_t.parameters()):
        p.requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=a.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    buf = ReplayBuffer(obs_dim, act_dim, min(a.buffer_size, a.total_steps))

    log = {"config": vars(a), "name": name, "evals": [], "train_episodes": []}
    n_critic_updates = 0
    critic_loss = actor_loss = q_batch = float("nan")
    t0 = time.time()
    s, _ = env.reset(seed=a.seed)
    ep_ret, ep_len = 0.0, 0

    for step in range(1, a.total_steps + 1):
        if step <= a.learning_starts:
            act = env.action_space.sample()
        else:
            with torch.no_grad():
                act = actor(torch.as_tensor(s, dtype=torch.float32)[None])[0].numpy()
            act = np.clip(act + rng.normal(0, a.expl_noise, act_dim), -1, 1).astype(np.float32)
        s2, r, done, trunc, _ = env.step(act)
        # Bootstrap through time-limit truncations, not through terminations.
        buf.add(s, act, r, s2, float(done))
        s = s2; ep_ret += r; ep_len += 1
        if done or trunc:
            log["train_episodes"].append((step, ep_ret, ep_len))
            s, _ = env.reset(); ep_ret, ep_len = 0.0, 0

        if step > a.learning_starts:
            for _ in range(a.utd):
                bs, ba, br, bs2, bd = buf.sample(rng, a.batch_size)
                with torch.no_grad():
                    a2 = actor_t(bs2)
                    if a.smooth:
                        noise = (torch.randn_like(a2) * a.target_noise).clamp(-a.noise_clip, a.noise_clip)
                        a2 = (a2 + noise).clamp(-1, 1)
                    q2 = critic_t(bs2, a2)
                    q2 = (a.beta * q2.min(1, keepdim=True).values
                          + (1 - a.beta) * q2.max(1, keepdim=True).values)
                    y = br + a.gamma * (1 - bd) * q2
                q = critic(bs, ba)
                critic_loss_t = F.mse_loss(q, y.expand_as(q)) * q.shape[1]  # sum over heads
                critic_opt.zero_grad(); critic_loss_t.backward(); critic_opt.step()
                n_critic_updates += 1

                if n_critic_updates % a.policy_delay == 0:
                    actor_loss_t = -critic(bs, actor(bs))[:, 0].mean()
                    actor_opt.zero_grad(); actor_loss_t.backward(); actor_opt.step()
                    with torch.no_grad():
                        for net, tgt in ((actor, actor_t), (critic, critic_t)):
                            for p, pt in zip(net.parameters(), tgt.parameters()):
                                pt.lerp_(p, a.tau)
                    actor_loss = actor_loss_t.item()
            critic_loss = critic_loss_t.item(); q_batch = q[:, 0].mean().item()

        if step % a.eval_every == 0:
            ev = evaluate(eval_env, actor, critic, a, eval_rng)
            ev.update(step=step, critic_loss=critic_loss, actor_loss=actor_loss,
                      q_batch=q_batch, time=time.time() - t0)
            log["evals"].append(ev)
            print(f"{name} step {step} ret {ev['return_mean']:.1f} "
                  f"bias {ev.get('bias', float('nan')):.2f} q {ev.get('q_mean', float('nan')):.1f} "
                  f"mc {ev.get('mc_mean', float('nan')):.1f} t {ev['time']:.0f}s", flush=True)
            with open(path + ".tmp", "w") as f:
                json.dump(log, f)
            os.replace(path + ".tmp", path)
            if a.save_ckpt:
                ckpts[step] = {"actor": {k: v.clone() for k, v in actor.state_dict().items()},
                               "critic": {k: v.clone() for k, v in critic.state_dict().items()}}
                torch.save({"config": vars(a), "ckpts": ckpts}, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)


if __name__ == "__main__":
    main()

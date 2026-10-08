"""Heatmaps of the saved critics' Q-values (Q1, the critic the actor follows).

Usage: python qgrid.py runs/showcase/*.pt [--step 300000] [--seed 0] [--out qgrid]

Two kinds of figures, one panel per agent (rows: LayerNorm off / on):
  qgrid_actions_<state>.png  Q1(s, a) over every action a = (main engine, side engines) for one
                             fixed state s, with the actor's action mu(s) marked. The states come
                             from one landing of a reference agent (standard TD3 without LayerNorm
                             if given), so every critic is queried on the same, realistic states.
  qgrid_positions.png        Q1(s, mu(s)) over lander positions (x, height) with the lander upright,
                             still and legs in the air: where each critic thinks it is good to be.
Colour scales differ between panels, because the critics' value ranges differ by up to 10x.
"""
import argparse
import os
from argparse import Namespace

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from td3 import Actor, Critic

BLUES = LinearSegmentedColormap.from_list(  # sequential: light = low Q, dark = high Q
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ACTOR, FLAG, INK, MUTED, GUIDE = "#eb6834", "#eda100", "#0b0b0b", "#52514e", "#9a9a96"
G = 101  # grid resolution per axis


def load(path, step):
    d = torch.load(path, map_location="cpu")
    cfg = Namespace(**d["config"])
    step = step if step is not None else max(d["ckpts"])
    env = gym.make(cfg.env)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    env.close()
    actor, critic = Actor(od, ad, cfg), Critic(od, ad, cfg, 2 if cfg.algo == "td3" else 1)
    actor.load_state_dict(d["ckpts"][step]["actor"])
    critic.load_state_dict(d["ckpts"][step]["critic"])
    beta = getattr(cfg, "beta", 1.0)
    name = "DDPG" if cfg.algo == "ddpg" else ("TD3" if beta == 1 else f"TD3 β={beta:g}")
    ln = "LayerNorm" if cfg.critic_ln else "no LayerNorm"
    return Namespace(actor=actor, critic=critic, cfg=cfg, title=f"{name} · {ln}", step=step,
                     col=(cfg.algo != "ddpg", beta), row=cfg.critic_ln)


@torch.no_grad()
def q1(agent, s, a):
    return agent.critic(torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(a, dtype=torch.float32))[:, 0].numpy()


@torch.no_grad()
def mu(agent, s):
    return agent.actor(torch.as_tensor(s, dtype=torch.float32)).numpy()


def reference_states(agent, seed):
    """Three states from the first episode (seed, seed+1, ...) in which the agent comes to rest."""
    env = gym.make(agent.cfg.env)
    for k in range(seed, seed + 50):
        s, _ = env.reset(seed=k)
        S, done, trunc, r = [s], False, False, 0.0
        while not (done or trunc):
            s, r, done, trunc, _ = env.step(mu(agent, s[None])[0])
            S.append(s)
        if done and r > 0:
            T = len(S) - 1
            return k, {"start": S[0], "mid-descent": S[T // 2], "before touchdown": S[max(0, T - 15)]}
    raise SystemExit(f"{agent.title} never landed in 50 episodes")


def grid_axes(agents, panel_w, panel_h):
    cols = sorted({ag.col for ag in agents})
    rows = sorted({ag.row for ag in agents})
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(panel_w * len(cols), panel_h * len(rows)),
                             squeeze=False)
    used = set()
    for ag in agents:
        ag.ax = axes[rows.index(ag.row), cols.index(ag.col)]
        used.add(id(ag.ax))
    for ax in axes.flat:
        if id(ax) not in used:
            ax.axis("off")
    return fig


def colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7, color=MUTED)
    cb.set_label(label, fontsize=7.5, color=MUTED)
    cb.outline.set_visible(False)


def action_grid(agents, s, label, out):
    a0, a1 = np.meshgrid(np.linspace(-1, 1, G), np.linspace(-1, 1, G))
    acts = np.stack([a0.ravel(), a1.ravel()], 1)
    fig = grid_axes(agents, 3.5, 3.15)
    for ag in agents:
        q = q1(ag, np.repeat(s[None], len(acts), 0), acts).reshape(G, G)
        m = mu(ag, s[None])[0]
        ax = ag.ax
        im = ax.imshow(q, origin="lower", extent=[-1, 1, -1, 1], cmap=BLUES, aspect="equal")
        colorbar(fig, im, ax, "Q₁(s, a)")
        ax.axvline(0, color=GUIDE, lw=0.8, ls="--")      # main engine off for a0 <= 0
        for v in (-0.5, 0.5):                              # side engines off for |a1| < 0.5
            ax.axhline(v, color=GUIDE, lw=0.8, ls="--")
        k = q.argmax()
        ax.scatter(a0.ravel()[k], a1.ravel()[k], marker="x", color=INK, s=36, lw=1.5, zorder=3, clip_on=False)
        ax.scatter(m[0], m[1], s=58, color=ACTOR, edgecolor="white", lw=1.5, zorder=4, clip_on=False)
        ax.set_title(f"{ag.title}\nQ at actor's action {q1(ag, s[None], m[None])[0]:.0f} · grid max {q.max():.0f}",
                     fontsize=8.5)
        ax.set_xlabel("main engine a₀", fontsize=8)
        ax.set_ylabel("side engines a₁", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=ACTOR, mec="white", label="actor's action μ(s)"),
               Line2D([], [], ls="", marker="x", ms=6, mew=1.5, color=INK, label="critic's maximum on the grid"),
               Line2D([], [], ls="--", color=GUIDE, label="engine switches (main off for a₀ ≤ 0, side off for |a₁| < 0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Q₁(s, a) over all actions, state “{label}”: x = {s[0]:+.2f}, height = {s[1]:.2f}, "
                 f"velocity = ({s[2]:+.2f}, {s[3]:+.2f}), angle = {s[4]:+.2f}  ·  colour scales differ between panels",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def position_grid(agents, start, out):
    X, Y = np.meshgrid(np.linspace(-1, 1, 121), np.linspace(0, 1.5, 91))
    S = np.zeros((X.size, 8), np.float32)  # upright, still, legs in the air
    S[:, 0], S[:, 1] = X.ravel(), Y.ravel()
    fig = grid_axes(agents, 3.9, 2.9)
    for ag in agents:
        v = q1(ag, S, mu(ag, S)).reshape(X.shape)
        ax = ag.ax
        im = ax.imshow(v, origin="lower", extent=[-1, 1, 0, 1.5], cmap=BLUES, aspect="auto")
        colorbar(fig, im, ax, "Q₁(s, μ(s))")
        for x in (-0.2, 0.2):  # the two flags of the landing pad
            ax.plot([x, x], [0, 0.12], color=FLAG, lw=2.5, solid_capstyle="butt")
        ax.scatter(start[0], start[1], s=40, facecolor="none", edgecolor=ACTOR, lw=1.8, zorder=3)
        ax.set_title(ag.title, fontsize=9)
        ax.set_xlabel("horizontal position x", fontsize=8)
        ax.set_ylabel("height above the pad", fontsize=8)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], color=FLAG, lw=2.5, label="landing pad flags (x = ±0.2)"),
               Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=ACTOR, mew=1.8, label="start position")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Q₁(s, μ(s)) at each position, lander upright, still, legs in the air  ·  "
                 "colour scales differ between panels", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+", help=".pt files written by td3.py --save-ckpt 1")
    p.add_argument("--step", type=int, default=None, help="snapshot to load (default: last)")
    p.add_argument("--seed", type=int, default=0, help="first episode seed for the reference landing")
    p.add_argument("--out", default="qgrid")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    agents = [load(path, a.step) for path in a.ckpts]
    ref = next((ag for ag in agents if ag.title == "TD3 · no LayerNorm"), agents[0])
    ep_seed, states = reference_states(ref, a.seed)
    print(f"reference states: landing of {ref.title} (episode seed {ep_seed})")
    for label, s in states.items():
        out = os.path.join(a.out, f"qgrid_actions_{label.replace(' ', '_')}.png")
        action_grid(agents, s, label, out)
        print("saved", out)
    out = os.path.join(a.out, "qgrid_positions.png")
    position_grid(agents, states["start"], out)
    print("saved", out)


if __name__ == "__main__":
    main()

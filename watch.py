"""Watch trained agents land, side by side, with their critic's estimate.

Usage:
  python watch.py runs/showcase/ddpg_*.pt runs/showcase/td3_*.pt            # live window
  python watch.py ... --step 150000 --episodes 5 --seed 3                  # earlier snapshot
  python watch.py ... --record landing.mp4 --episodes 3                   # save a video (ffmpeg)
  python watch.py a.pt b.pt c.pt d.pt e.pt f.pt --cols 3                   # 2 x 3 grid
  python watch.py noop runs/showcase/td3_ln10_*.pt                        # "do nothing" baseline

All agents fly the same episodes (same terrain and initial push). Below each
lander, the chart shows the critic's Q(s_t, a_t) as the episode unfolds; when the
episode ends, the true discounted return G_t of that episode is drawn over it.
The gap between the two curves is the overestimation the report measures.
Keys: space = pause, n = next episode, q / Esc = quit.
"""
import argparse
import os
import subprocess
from argparse import Namespace

import gymnasium as gym
import numpy as np
import pygame
import torch

from td3 import Actor, Critic

W, H_ENV, H_CHART, H_HEAD = 600, 400, 230, 52
BG, INK, MUTED, GRID = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 228, 224)
Q_COLOR, G_COLOR = (235, 104, 52), (42, 120, 214)  # orange = critic, blue = MC return


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+",
                   help=".pt files written by td3.py --save-ckpt 1, or 'noop' for an agent that always plays [0, 0]")
    p.add_argument("--step", type=int, default=None, help="snapshot to load (default: last)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--record", default=None, help="write an mp4 instead of only showing")
    p.add_argument("--headless", action="store_true", help="no window (with --record)")
    p.add_argument("--cols", type=int, default=None, help="panels per row (default: all in one row)")
    p.add_argument("--scale", type=float, default=None, help="window scale (default: fit the screen)")
    return p.parse_args()


class Agent:
    def __init__(self, path, step):
        self.critic = None
        if path == "noop":  # baseline: both engines always off, no critic
            self.actor, self.gamma = None, 0.99
            self.title = "Do nothing · action always [0, 0]"
            self.env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
            return
        d = torch.load(path, map_location="cpu")
        cfg = Namespace(**d["config"])
        step = step if step is not None else max(d["ckpts"])
        if step not in d["ckpts"]:
            raise SystemExit(f"{path}: no snapshot at step {step}; have {sorted(d['ckpts'])}")
        env = gym.make(cfg.env)
        od, ad = env.observation_space.shape[0], env.action_space.shape[0]
        env.close()
        self.actor = Actor(od, ad, cfg)
        self.critic = Critic(od, ad, cfg, 2 if cfg.algo == "td3" else 1)
        self.actor.load_state_dict(d["ckpts"][step]["actor"])
        self.critic.load_state_dict(d["ckpts"][step]["critic"])
        ln = "critic LayerNorm" if cfg.critic_ln else "no LayerNorm"
        beta = getattr(cfg, "beta", 1.0)
        algo = cfg.algo.upper() + ("" if beta == 1 else f" β={beta:g}")
        self.title = f"{algo} · {ln} · seed {cfg.seed} · {step // 1000}k steps"
        self.gamma = cfg.gamma
        self.env = gym.make(cfg.env, render_mode="rgb_array")

    def reset(self, seed):
        self.s, _ = self.env.reset(seed=seed)
        self.qs, self.rewards, self.done, self.ret, self.outcome = [], [], False, 0.0, ""
        self.frame = self.env.render()

    @torch.no_grad()
    def step(self):
        if self.done:
            return
        if self.actor is None:
            act = np.zeros(2, np.float32)
        else:
            s = torch.as_tensor(self.s, dtype=torch.float32)[None]
            a_t = self.actor(s)
            self.qs.append(self.critic(s, a_t)[0, 0].item())
            act = a_t[0].numpy()
        self.s, r, term, trunc, _ = self.env.step(act)
        self.rewards.append(r); self.ret += r
        self.frame = self.env.render()
        if term or trunc:
            self.done = True
            x = self.s[0]  # 0 = pad centre, the pad spans |x| <= 0.2
            if trunc:
                self.outcome = "time limit"
            elif r > 0:  # +100: came to rest, wherever that is
                self.outcome = f"landed {'on' if abs(x) <= 0.2 else 'off'} pad, x={x:+.2f}"
            else:
                self.outcome = "left screen" if abs(x) >= 1 else "crashed"
            g, self.G = 0.0, np.zeros(len(self.rewards))
            for t in reversed(range(len(self.rewards))):
                g = self.rewards[t] + self.gamma * g
                self.G[t] = g


def draw_agent(surf, x0, y0, ag, fonts, t_max):
    big, small = fonts
    # header
    surf.blit(big.render(ag.title, True, INK), (x0 + 12, y0 + 8))
    status = f"t={len(ag.rewards):4d}   return={ag.ret:7.1f}"
    if ag.qs:
        status += f"   Q(s,a)={ag.qs[-1]:7.1f}"
    if ag.done:
        status += f"   [{ag.outcome}]"
    surf.blit(small.render(status, True, MUTED), (x0 + 12, y0 + 30))
    # environment frame
    surf.blit(pygame.surfarray.make_surface(ag.frame.swapaxes(0, 1)), (x0, y0 + H_HEAD))
    # chart
    cy, ch, cx, cw = y0 + H_HEAD + H_ENV + 22, H_CHART - 46, x0 + 48, W - 64
    series = [np.array(ag.qs)] + ([ag.G] if ag.done else [])
    vals = np.concatenate([s for s in series if len(s)] or [np.zeros(1)])
    lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
    pad = 0.1 * (hi - lo) + 1.0
    lo, hi = lo - pad, hi + pad
    ty = lambda v: cy + ch - (v - lo) / (hi - lo) * ch
    tx = lambda t: cx + t / t_max * cw
    for v in np.linspace(lo, hi, 5):
        pygame.draw.line(surf, GRID, (cx, ty(v)), (cx + cw, ty(v)))
        surf.blit(small.render(f"{v:.0f}", True, MUTED), (x0 + 4, ty(v) - 8))
    pygame.draw.line(surf, MUTED, (cx, ty(0)), (cx + cw, ty(0)))
    for s, color in zip(series, (Q_COLOR, G_COLOR)):
        if len(s) > 1:
            pygame.draw.lines(surf, color, False, [(tx(t), ty(v)) for t, v in enumerate(s)], 2)
    legend = ([("critic Q(s_t, a_t)", Q_COLOR)] if ag.critic is not None else []) + [
        ("true return G_t (at episode end)", G_COLOR)]
    lx = cx
    for text, color in legend:
        pygame.draw.line(surf, color, (lx, cy - 12), (lx + 18, cy - 12), 3)
        img = small.render(text, True, INK)
        surf.blit(img, (lx + 24, cy - 20)); lx += img.get_width() + 44
    if ag.done and len(ag.G) and ag.qs:
        # same rule as td3.py: G is incomplete in the last 500 steps of a time-limit episode
        keep = len(ag.G) if ag.outcome != "time limit" else len(ag.G) - 500
        if keep > 0:
            gap = np.mean(np.array(ag.qs[:keep]) - ag.G[:keep])
            note = "" if keep == len(ag.G) else f" (first {keep} steps)"
            surf.blit(small.render(f"mean Q − G over episode{note}: {gap:+.1f}", True, INK),
                      (cx, cy + ch + 6))


def main():
    a = parse_args()
    if a.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    agents = [Agent(p, a.step) for p in a.ckpts]
    pygame.init()
    cols = a.cols or len(agents)
    rows = -(-len(agents) // cols)
    panel_h = H_HEAD + H_ENV + H_CHART
    size = (W * cols, panel_h * rows)
    canvas = pygame.Surface(size)  # full resolution, also what gets recorded
    scale = a.scale or 1.0
    if not a.headless and a.scale is None:
        dw, dh = pygame.display.get_desktop_sizes()[0]
        scale = min(1.0, 0.95 * dw / size[0], 0.88 * dh / size[1])
    win = (int(size[0] * scale), int(size[1] * scale))
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("LunarLander — critic estimate vs true return")
    fonts = (pygame.font.SysFont("dejavusans", 15, bold=True), pygame.font.SysFont("dejavusansmono", 13))
    clock = pygame.time.Clock()
    ff = None
    if a.record:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{size[0]}x{size[1]}", "-r", str(a.fps), "-i", "-",
                               "-pix_fmt", "yuv420p", "-vcodec", "libx264", a.record], stdin=subprocess.PIPE)
    rng = np.random.default_rng(a.seed)
    try:
        for ep in range(a.episodes):
            seed = int(rng.integers(2**31))
            for ag in agents:
                ag.reset(seed)
            paused, hold, skip = False, 0, False
            while hold < a.fps * 2 and not skip:  # keep the final frame 2 s
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                        return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                        paused = not paused
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                        skip = True
                if paused:
                    clock.tick(a.fps); continue
                for ag in agents:
                    ag.step()
                if all(ag.done for ag in agents):
                    hold += 1
                t_max = max(200, *(len(ag.rewards) for ag in agents))
                canvas.fill(BG)
                for i, ag in enumerate(agents):
                    draw_agent(canvas, (i % cols) * W, (i // cols) * panel_h, ag, fonts, t_max)
                for c in range(1, cols):
                    pygame.draw.line(canvas, GRID, (c * W, 0), (c * W, size[1]), 2)
                for r in range(1, rows):
                    pygame.draw.line(canvas, GRID, (0, r * panel_h), (size[0], r * panel_h), 2)
                screen.blit(canvas if win == size else pygame.transform.smoothscale(canvas, win), (0, 0))
                pygame.display.flip()
                if ff:
                    ff.stdin.write(pygame.surfarray.array3d(canvas).swapaxes(0, 1).tobytes())
                if not a.headless:
                    clock.tick(a.fps)
            for ag in agents:
                print(f"episode {ep} | {ag.title} | return {ag.ret:.1f} ({ag.outcome})")
    finally:
        if ff:
            ff.stdin.close(); ff.wait()
            print("saved", a.record)
        pygame.quit()


if __name__ == "__main__":
    main()

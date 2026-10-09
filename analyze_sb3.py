"""Statistics of the Stable-Baselines3 replication (mp-sb3), in the form of the main results table.

Usage: python analyze_sb3.py [mp-sb3/results/core] [--out figs_extra]

Per run, as defined in the SB3 code: final return = mean of the last 5 evaluations, success = share
of those 50 episodes with return >= 200, bias = mean bias over the same evaluations.
Comparisons as in Table 2 (TD3 - DDPG per LayerNorm setting, LayerNorm x algorithm interaction,
LayerNorm effect per algorithm) for the 3 metrics: Welch-Satterthwaite p-values with a Holm
correction over the 15 tests. With 5 seeds per condition the bootstrap CIs are too narrow, so none are
reported; the t-based CIs are in the SB3 code's own output (mp-sb3/figures/stats_contrasts.csv).
The per-episode evaluation returns come from each run's eval_returns.csv, extracted from the raw
eval_raw/*.npz files once the run has finished (only the CSV is in the repository).
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from analyze import SUCCESS, contrast, holm

ALGOS = {"ddpg": "DDPG", "td3": "TD3"}
METRICS = (("final return", 0, 1), ("success (%)", 1, 100), ("bias", 2, 1))
COMPARISONS = (
    ("TD3 - DDPG, LN off", [(("td3", 0), 1), (("ddpg", 0), -1)]),
    ("TD3 - DDPG, LN on", [(("td3", 1), 1), (("ddpg", 1), -1)]),
    ("LN x algorithm", [(("td3", 1), 1), (("td3", 0), -1), (("ddpg", 1), -1), (("ddpg", 0), 1)]),
    ("LN effect, DDPG", [(("ddpg", 1), 1), (("ddpg", 0), -1)]),
    ("LN effect, TD3", [(("td3", 1), 1), (("td3", 0), -1)]),
)


def eval_returns(run_dir, steps):
    """Per-episode returns of each evaluation (one array per step), cached in eval_returns.csv."""
    cache = f"{run_dir}/eval_returns.csv"
    if not os.path.exists(cache):
        eps = {s: np.load(f"{run_dir}/eval_raw/step_{s:08d}.npz")["returns"] for s in steps}
        if not os.path.exists(f"{run_dir}/summary.json"):  # still training: no cache yet
            return [eps[s] for s in steps]
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "return"])
            w.writerows((s, float(r)) for s in steps for r in eps[s])
    by_step = defaultdict(list)
    for row in csv.DictReader(open(cache)):
        by_step[int(row["step"])].append(float(row["return"]))
    return [np.array(by_step[s]) for s in steps]


def load_sb3(folder):
    """{(algo, ln): [run]}; a run holds its evaluation steps, mean returns, episode returns and bias."""
    groups = defaultdict(list)
    for cfg in sorted(glob.glob(f"{folder}/algo=*")):
        algo, ln = cfg.split("algo=")[1].split(",")[0], int(cfg.split("critic_ln=")[1][0])
        for run_dir in sorted(glob.glob(f"{cfg}/seed*")):
            if not os.path.exists(f"{run_dir}/evals.csv"):
                continue
            rows = list(csv.DictReader(open(f"{run_dir}/evals.csv")))
            steps = [int(r["step"]) for r in rows]
            groups[(algo, ln)].append(dict(
                total=json.load(open(f"{run_dir}/config.json"))["total_steps"], n_final=5, bias_half=False,
                steps=np.array(steps), ret=np.array([float(r["return_mean"]) for r in rows]),
                episodes=eval_returns(run_dir, steps), bias=np.array([float(r["bias"] or "nan") for r in rows])))
    return groups


def at(run, upto):
    """(final return, success, bias) of a run as if it had stopped at step `upto`; None if not reached."""
    idx = np.flatnonzero(run["steps"] <= upto)
    if not len(idx) or run["steps"][idx[-1]] != upto:
        return None
    last = idx[-run["n_final"]:]
    success = np.mean(np.concatenate([run["episodes"][i] for i in last]) >= SUCCESS)
    if run["bias_half"]:
        bias = np.nanmean(run["bias"][(run["steps"] >= upto / 2) & (run["steps"] <= upto)])
    else:
        bias = np.nanmean(run["bias"][last])
    return run["ret"][last].mean(), success, bias


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="mp-sb3/results/core")
    ap.add_argument("--out", default="figs_extra")
    a = ap.parse_args()
    groups = load_sb3(a.folder)
    total = max(r["total"] for runs in groups.values() for r in runs)
    summ = {k: np.array([s for s in (at(r, total) for r in runs) if s is not None]) for k, runs in groups.items()}
    text = [f"SB3 replication ({a.folder}, {total // 1000}k steps)"]
    for (algo, ln), s in sorted(summ.items()):
        text.append(f"  {ALGOS[algo]:4s} LN {'on ' if ln else 'off'} n={len(s)}  " + "  ".join(
            f"{name} {s[:, k].mean() * scale:6.1f}" for name, k, scale in METRICS))
    rows = [(f"{label:20s} {name:12s}: {contrast(summ, terms, k)[0] * scale:+7.1f}", contrast(summ, terms, k)[3])
            for label, terms in COMPARISONS for name, k, scale in METRICS]
    for (line, p), adj in zip(rows, holm([p for _, p in rows])):
        text.append(f"  {line}  p={p:.3g}  Holm p={adj:.3g}{'  †' if adj < 0.05 else ''}")
    os.makedirs(a.out, exist_ok=True)
    name = "sb3_stats.txt" if total == 150_000 else f"sb3_{total // 1000}k_stats.txt"
    open(f"{a.out}/{name}", "w").write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

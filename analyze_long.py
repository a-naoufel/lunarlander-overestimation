"""Analysis G (ANALYSIS_PLAN.md): does DDPG reach a return of 200 when trained longer?

Usage: python analyze_long.py [--ours runs/long] [--sb3 mp-sb3/results_300k/core] [--out figs_extra]

Our code: DDPG / TD3 x critic LayerNorm x 10 seeds, 600k steps (twice the main budget).
SB3 code (mp-sb3): its core grid (5 seeds) with 300k steps instead of 150k.
Per run, as defined in each code base: final return = mean of the last evaluations (3 in our code,
5 in the SB3 code), success = share of those episodes with return >= 200, bias (our code: second
half of training; SB3 code: the final evaluations). "Half" = the same metrics computed at half of
the training (300k / 150k steps) from the same runs.
Tests, per code base: end vs. half per condition (paired t-test, bootstrap CI of the mean paired
difference) and TD3 - DDPG at the end per LayerNorm setting (bootstrap CI, Welch's t-test), with a
Holm correction over these 6 tests. Runs still training are drawn but left out of the tests.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze import GRID, GUIDE, LN_TITLES, MUTED, PAGE_W, SUCCESS, VARIANT_COLORS, bootstrap_ci, contrast, holm
from analyze_sb3 import ALGOS, at, load_sb3

CODES = (("ours", "Our code"), ("sb3", "SB3 code"))


def load_ours(folder):
    """{(algo, ln): [run]}; a run holds its evaluation steps, mean returns, episode returns and bias."""
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{folder}/*.json")):
        d = json.load(open(f))
        c, ev = d["config"], d["evals"]
        groups[(c["algo"], int(c["critic_ln"]))].append(dict(
            total=c["total_steps"], n_final=3, bias_half=True, steps=np.array([e["step"] for e in ev]),
            ret=np.array([e["return_mean"] for e in ev]), episodes=[np.array(e["returns"]) for e in ev],
            bias=np.array([e["bias"] for e in ev])))
    return groups


def analyse(groups, title, text):
    total = max(r["total"] for runs in groups.values() for r in runs)
    half = total // 2
    text.append(f"{title}: {total // 1000}k steps; 'half' = the same metrics at {half // 1000}k steps")
    rows, ps, summ = [], [], {}
    for (algo, ln) in sorted(groups):
        runs = groups[(algo, ln)]
        pairs = [(at(r, total), at(r, half)) for r in runs]
        pairs = [(e, h) for e, h in pairs if e is not None and h is not None]
        if len(pairs) < 2:
            text.append(f"  {ALGOS[algo]} LN {'on' if ln else 'off'}: {len(pairs)} finished runs of {len(runs)}")
            continue
        end, hal = np.array([e for e, _ in pairs]), np.array([h for _, h in pairs])
        summ[(algo, ln)] = end
        d = end[:, 0] - hal[:, 0]
        lo, hi = bootstrap_ci(d)
        p = stats.ttest_rel(end[:, 0], hal[:, 0]).pvalue
        ever = sum(r["ret"].max() >= SUCCESS for r in runs if at(r, total) is not None)
        rows.append([f"  {ALGOS[algo]:4s} LN {'on ' if ln else 'off'} n={len(end):2d}  return at {half // 1000}k "
                     f"{hal[:, 0].mean():6.1f} -> at {total // 1000}k {end[:, 0].mean():6.1f}  change {d.mean():+6.1f} "
                     f"[{lo:+.1f}, {hi:+.1f}] p={p:.3g}", f"| success {end[:, 1].mean() * 100:4.1f}%  "
                     f"bias {end[:, 2].mean():6.1f}  runs ending >= {SUCCESS}: {(end[:, 0] >= SUCCESS).sum()}/{len(end)}"
                     f"  runs with an evaluation >= {SUCCESS}: {ever}/{len(end)}"])
        ps.append(p)
    for ln in (0, 1):
        if ("td3", ln) in summ and ("ddpg", ln) in summ:
            est, lo, hi, p = contrast(summ, [(("td3", ln), 1), (("ddpg", ln), -1)], 0)
            rows.append([f"  TD3 - DDPG at {total // 1000}k, LN {'on ' if ln else 'off'}: {est:+6.1f} "
                         f"[{lo:+.1f}, {hi:+.1f}] p={p:.3g}", ""])
            ps.append(p)
    for row, adj in zip(rows, holm(ps) if ps else []):
        text.append(f"{row[0]} Holm p={adj:.3g} {row[1]}".rstrip())
    return summ, total


def mean_curve(runs):
    """Mean evaluation return and 95% bootstrap CI over runs, up to the last step all runs have reached."""
    n = min(len(r["steps"]) for r in runs)
    rets = np.array([r["ret"][:n] for r in runs])
    return runs[0]["steps"][:n], rets.mean(0), np.array([bootstrap_ci(rets[:, i]) for i in range(n)])


def plot(data, out):
    fig, axes = plt.subplots(len(data), 2, figsize=(PAGE_W, 2.35 * len(data) + 0.35), sharey="row", squeeze=False)
    for i, (title, groups, summ, total) in enumerate(data):
        for j, ln in enumerate((0, 1)):
            ax = axes[i, j]
            ax.axhline(0, color=GUIDE, lw=0.6)
            ax.axhline(SUCCESS, color=GUIDE, lw=0.6, ls=":")
            ax.axvline(total / 2e3, color=GUIDE, lw=0.8, ls="--")
            counts = []
            for c, algo in enumerate(ALGOS):
                runs = groups.get((algo, ln))
                if not runs:
                    continue
                steps, mean, ci = mean_curve(runs)
                ax.plot(steps / 1e3, mean, color=VARIANT_COLORS[c], lw=1.6, label=ALGOS[algo])
                ax.fill_between(steps / 1e3, ci[:, 0], ci[:, 1], color=VARIANT_COLORS[c], alpha=0.18, lw=0)
                if (algo, ln) in summ:
                    end = summ[(algo, ln)][:, 0]
                    counts.append(f"{ALGOS[algo]} {(end >= SUCCESS).sum()}/{len(end)}")
            ax.set_title(f"{title}, {LN_TITLES[ln]}", fontweight="bold", fontsize=9, pad=14)
            if counts:
                ax.text(0.5, 1.015, f"runs ending ≥ {SUCCESS}: " + ", ".join(counts), transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7.5, color=MUTED)
            ax.set_xlim(0, total / 1e3)
            ax.set_ylabel("Evaluation return")
            ax.set_xlabel("Environment steps (×1000)")
            ax.tick_params(labelleft=True)
            ax.grid(axis="y", color=GRID, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="runs/long")
    ap.add_argument("--sb3", default="mp-sb3/results_300k/core")
    ap.add_argument("--out", default="figs_extra")
    a = ap.parse_args()
    text, data = [], []
    for (code, title), groups in zip(CODES, (load_ours(a.ours), load_sb3(a.sb3))):
        if groups:
            summ, total = analyse(groups, title, text)
            data.append((title, groups, summ, total))
    os.makedirs(a.out, exist_ok=True)
    plot(data, f"{a.out}/long.png")
    open(f"{a.out}/long_stats.txt", "w").write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

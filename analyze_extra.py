"""The four additional analyses of ANALYSIS_PLAN.md (exploratory).

Usage: python analyze_extra.py [--out figs_extra]
  A. timing: does the early bias (50k-150k) predict the later improvement of the return?
  B. bias vs final return within conditions (z-scores within each condition)
  C. dose-response: TD3 with target weight beta in {0, 0.25, 0.5, 0.75, 1}   (runs/dose)
  D. ablation: DDPG + one TD3 change (twin critics / smoothing / delay)      (runs/ablation)
A and B use the main runs (runs/core, runs/beta075); C and D are skipped until their
runs exist. Statistics are those of analyze.py (bootstrap CIs, Welch tests, Holm).
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

from analyze import (GRID, GUIDE, MINUS, MUTED, PAGE_W, bootstrap_ci, contrast, fmt, fmt_p, holm,
                     summarize)

DDPG_COLOR, TD3_COLOR = "#2a78d6", "#eb6834"  # as in the report's figures
METRICS = ((0, "final return", 1), (2, "success (%)", 100), (1, "bias", 1))
N_PERM = 10_000


def load_runs(dirs):
    """{(variant, ln): [run, ...]} with variant = (algo, twin, smooth, policy_delay, beta)."""
    groups = defaultdict(list)
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            run = json.load(open(f))
            c = run["config"]
            td3 = int(c["algo"] == "td3")
            variant = (c["algo"], c.get("twin", td3), c.get("smooth", td3), c["policy_delay"], c.get("beta", 1.0))
            groups[(variant, c["critic_ln"])].append(run)
    return groups


def variant_name(v):
    algo, twin, smooth, pd, beta = v
    if algo == "td3":
        return "TD3" if beta == 1 else f"TD3, β = {beta:g}"
    extra = [n for flag, n in ((twin, "twin critics"), (smooth, "target smoothing"), (pd > 1, "delayed updates")) if flag]
    return "DDPG" + "".join(f" + {n}" for n in extra)


DDPG = ("ddpg", 0, 0, 1, 1.0)
TD3 = ("td3", 1, 1, 2, 1.0)
ABLATIONS = (("ddpg", 0, 0, 2, 1.0), ("ddpg", 0, 1, 1, 1.0), ("ddpg", 1, 0, 1, 1.0))  # delay, smoothing, twin
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def summaries(groups):
    summ = {}
    for k, runs in groups.items():
        s = [x for x in map(summarize, runs) if x is not None]
        if s:
            summ[k] = s
    return summ


def mean_between(run, key, lo, hi):
    return np.mean([e[key] for e in run["evals"] if lo <= e["step"] <= hi and key in e])


def stratified_corr(xs, ys, groups, n_perm=N_PERM, seed=0):
    """Pearson correlation of within-group standardized values, with a within-group permutation p-value."""
    xs, ys, groups = np.asarray(xs, float), np.asarray(ys, float), np.asarray(groups)
    zx, zy = np.empty_like(xs), np.empty_like(ys)
    for g in np.unique(groups):
        m = groups == g
        zx[m] = (xs[m] - xs[m].mean()) / xs[m].std()
        zy[m] = (ys[m] - ys[m].mean()) / ys[m].std()
    r = np.mean(zx * zy)
    rng = np.random.default_rng(seed)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    count = 0
    for _ in range(n_perm):
        perm = np.empty(len(zy))
        for ix in idx:
            perm[ix] = zy[rng.permutation(ix)]
        count += abs(np.mean(zx * perm)) >= abs(r)
    return r, (count + 1) / (n_perm + 1)


def ranks_within(values, groups):
    values, groups = np.asarray(values, float), np.asarray(groups)
    out = np.empty_like(values)
    for g in np.unique(groups):
        m = groups == g
        out[m] = stats.rankdata(values[m])
    return out


def analysis_a(groups, text):
    """Early bias (50k-150k) vs later improvement (final - return at 140k-160k)."""
    rows = []
    for (variant, ln), runs in groups.items():
        for run in runs:
            s = summarize(run)
            if s is None:
                continue
            early = mean_between(run, "bias", 50_000, 150_000)
            improvement = s[0] - mean_between(run, "return_mean", 140_000, 160_000)
            rows.append((variant, ln, early, improvement))
    text.append("A. Early bias (50k-150k) vs later improvement (final - return at 140k-160k);"
                " Spearman on within-condition ranks, permutation p")
    cond = [f"{variant_name(v)}|{ln}" for v, ln, _, _ in rows]
    rx, ry = ranks_within([r[2] for r in rows], cond), ranks_within([r[3] for r in rows], cond)
    rho, p = stratified_corr(rx, ry, cond)
    text.append(f"   all conditions (n = {len(rows)}): rho = {rho:+.2f}, p = {p:.3g}")
    out = {}
    for v in sorted({r[0] for r in rows}, key=variant_name):
        m = [i for i, r in enumerate(rows) if r[0] == v]
        rho_v, p_v = stratified_corr(rx[m], ry[m], np.array(cond)[m])
        text.append(f"   {variant_name(v):14s} (n = {len(m)}): rho = {rho_v:+.2f}, p = {p_v:.3g}")
        out[variant_name(v)] = (rho_v, p_v)
    out["all"] = (rho, p)
    return out


def analysis_b(summ, text):
    """Final return vs bias, z-scored within each condition."""
    xs, ys, cond = [], [], []
    for k, s in summ.items():
        for final, bias, _ in s:
            xs.append(bias); ys.append(final); cond.append(f"{variant_name(k[0])}|{k[1]}")
    r, p = stratified_corr(xs, ys, cond)
    text.append(f"B. Bias vs final return within conditions (z-scores, n = {len(xs)}): r = {r:+.2f}, p = {p:.3g}")
    return r, p


def family(summ, comparisons, text, title):
    """Contrasts (a - b) for every metric, with a Holm correction over the whole family."""
    res = [[contrast(summ, [(a, 1), (b, -1)], k) for k, _, _ in METRICS] for _, a, b in comparisons]
    adj = holm([r[3] for row in res for r in row]).reshape(len(comparisons), len(METRICS))
    text.append(title)
    lines = []
    for (label, _, _), row, row_adj in zip(comparisons, res, adj):
        cells = []
        for (est, lo, hi, p), pa, (_, metric, scale) in zip(row, row_adj, METRICS):
            cells.append(f"{fmt(est, lo, hi, scale)}, {fmt_p(p)}" + (r"$^\dagger$" if pa < 0.05 else ""))
            plain = label.replace("$\\beta", "β").replace("$-$", "-").replace("$", "")
            text.append(f"   {plain:38s} {metric:13s}: {scale * est:+7.1f} [{scale * lo:7.1f}, {scale * hi:7.1f}]"
                        f"  p={p:.3g}  Holm p={pa:.3g}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return lines, adj


def table(lines, out):
    head = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
            r"Comparison & Final return & Success (\%) & Bias \\", r"\midrule"]
    with open(out, "w") as fh:
        fh.write("\n".join(head + [l.replace("β", r"$\beta$") for l in lines] + [r"\bottomrule", r"\end{tabular}"]) + "\n")


def analysis_c(summ, out_dir, text):
    keys = {(b, ln): (("td3", 1, 1, 2, b), ln) for b in BETAS for ln in (0, 1)}
    if not all(k in summ for k in keys.values()):
        text.append("C. dose-response: runs not available yet")
        return
    text.append("C. Dose-response on beta (TD3): Spearman between beta and each per-run metric (n = 50 per LN setting)")
    for ln in (0, 1):
        for k, metric, scale in METRICS:
            b = [beta for beta in BETAS for _ in summ[keys[(beta, ln)]]]
            y = [s[k] for beta in BETAS for s in summ[keys[(beta, ln)]]]
            rho, p = stats.spearmanr(b, y)
            text.append(f"   LN {'on ' if ln else 'off'} {metric:13s}: rho = {rho:+.2f}, p = {p:.3g}")
    comps = [(f"$\\beta = {b:g}$ $-$ TD3, LN {'on' if ln else 'off'}", keys[(b, ln)], keys[(1.0, ln)])
             for ln in (0, 1) for b in (0.5, 0.25, 0.0)]
    lines, adj = family(summ, comps, text, "   each beta vs beta = 1 (Holm over 18 tests):")
    table(lines, os.path.join(out_dir, "dose_table.tex"))
    sig = {(a, k): adj[i, j] < 0.05 for i, (_, a, _) in enumerate(comps) for j, (k, _, _) in enumerate(METRICS)}
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.6), squeeze=False)
    for j, (k, metric, scale) in enumerate(METRICS):
        ax = axes[0, j]
        for ln, (ls, face) in ((0, ("--", "white")), (1, ("-", TD3_COLOR))):
            m, lo, hi = zip(*[(np.mean([s[k] for s in summ[keys[(b, ln)]]]), *bootstrap_ci([s[k] for s in summ[keys[(b, ln)]]]))
                              for b in BETAS])
            m, lo, hi = scale * np.array(m), scale * np.array(lo), scale * np.array(hi)
            ax.errorbar(BETAS, m, yerr=[m - lo, hi - m], color=TD3_COLOR, ls=ls, lw=1.4, marker="o", ms=5,
                        mfc=face, mec=TD3_COLOR, capsize=2.5, label="critic LayerNorm" if ln else "no LayerNorm")
        if k == 1:
            ax.axhline(0, color=GUIDE, lw=0.6)
        ax.set_xlabel("TD3 target weight β")
        ax.set_ylabel({"final return": "Final return", "success (%)": "Success rate (%)", "bias": "Bias (150k–300k)"}[metric])
        ax.set_xticks(BETAS)
        ax.grid(axis="y", color=GRID, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out_dir, "dose.png"), dpi=300)
    plt.close(fig)
    return sig


def analysis_d(summ, out_dir, text):
    names = (DDPG, *ABLATIONS, TD3)
    if not all((v, ln) in summ for v in names for ln in (0, 1)):
        text.append("D. ablation: runs not available yet")
        return
    comps = [(f"{variant_name(v)} $-$ DDPG, LN {'on' if ln else 'off'}", (v, ln), (DDPG, ln))
             for ln in (0, 1) for v in ABLATIONS]
    lines, adj = family(summ, comps, text, "D. Ablation: DDPG + one TD3 change vs DDPG (Holm over 18 tests):")
    table(lines, os.path.join(out_dir, "ablation_table.tex"))
    sig = {(a, k): adj[i, j] < 0.05 for i, (_, a, _) in enumerate(comps) for j, (k, _, _) in enumerate(METRICS)}
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.5), sharey=True, squeeze=False)
    ypos = np.arange(len(names))[::-1]
    for j, (k, metric, scale) in enumerate(METRICS):
        ax = axes[0, j]
        for ln, dy, face in ((0, 0.13, "white"), (1, -0.13, None)):
            for y, v in zip(ypos, names):
                vals = [s[k] for s in summ[(v, ln)]]
                m, (lo, hi) = scale * np.mean(vals), scale * np.array(bootstrap_ci(vals))
                color = TD3_COLOR if v == TD3 else DDPG_COLOR
                ax.errorbar(m, y + dy, xerr=[[m - lo], [hi - m]], color=color, marker="o", ms=5, lw=1.2, capsize=2,
                            mfc=face or color, mec=color)
        if k == 1:
            ax.axvline(0, color=GUIDE, lw=0.6)
        ax.set_xlabel({"final return": "Final return", "success (%)": "Success rate (%)", "bias": "Bias (150k–300k)"}[metric])
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels([variant_name(v).replace("DDPG + ", "+ ") for v in names])
        ax.tick_params(labelleft=True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MUTED, label="no LayerNorm"),
               Line2D([], [], ls="", marker="o", ms=5, mfc=MUTED, mec=MUTED, label="critic LayerNorm")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out_dir, "ablation.png"), dpi=300)
    plt.close(fig)
    return sig


AXIS_LABELS = {"final return": "Final return", "success (%)": "Success rate (%)", "bias": "Bias (150k–300k)"}
SHORT = {DDPG: "DDPG", ABLATIONS[0]: "+DP", ABLATIONS[1]: "+TPS", ABLATIONS[2]: "+CDQ", TD3: "TD3"}


def dagger(ax, x, top, color):
    """Mark a point whose difference with its reference survives the Holm correction."""
    ax.annotate("†", (x, top), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                fontsize=8.5, color=color)


def plot_controlled(summ, sig, out):
    """Report figure: (a) dose-response on beta, (b) ablation; one column per metric, y shared per column."""
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, 4.4), sharey="col", squeeze=False)
    names = (DDPG, *ABLATIONS, TD3)
    for j, (k, metric, scale) in enumerate(METRICS):
        ax = axes[0, j]
        for ln, ls, face, dx in ((0, "--", "white", -0.025), (1, "-", TD3_COLOR, 0.025)):  # side by side
            keys = [(("td3", 1, 1, 2, b), ln) for b in BETAS]
            vals = [[s[k] for s in summ[key]] for key in keys]
            m = scale * np.array([np.mean(v) for v in vals])
            lo, hi = (scale * np.array(x) for x in zip(*map(bootstrap_ci, vals)))
            xs = np.array(BETAS) + dx
            ax.errorbar(xs, m, yerr=[m - lo, hi - m], color=TD3_COLOR, ls=ls, lw=1.3, marker="o", ms=4.5,
                        mfc=face, mec=TD3_COLOR, capsize=2)
            for x, key, top in zip(xs, keys, hi):
                if sig.get((key, k)):
                    dagger(ax, x, top, TD3_COLOR)
        ax.set_xticks(BETAS)
        ax.set_xlabel("TD3 target weight β")
        ax = axes[1, j]
        for ln, dx in ((0, -0.15), (1, 0.15)):
            for x, v in enumerate(names):
                vals = [s[k] for s in summ[(v, ln)]]
                m, (lo, hi) = scale * np.mean(vals), scale * np.array(bootstrap_ci(vals))
                color = TD3_COLOR if v == TD3 else DDPG_COLOR
                ax.errorbar(x + dx, m, yerr=[[m - lo], [hi - m]], color=color, marker="o", ms=4.5, lw=1.2,
                            capsize=2, mfc=color if ln else "white", mec=color)
                if sig.get(((v, ln), k)):
                    dagger(ax, x + dx, hi, color)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[v] for v in names], fontsize=7.2)
        for row in (0, 1):
            a = axes[row, j]
            if k == 1:
                a.axhline(0, color=GUIDE, lw=0.6)
            a.set_ylabel(AXIS_LABELS[metric])
            a.tick_params(labelleft=True)
            a.grid(axis="y", color=GRID, lw=0.5)
    axes[0, 1].set_title("(a) Dose-response: TD3 with target weight β", fontweight="bold", fontsize=9, pad=8)
    axes[1, 1].set_title("(b) Ablation: DDPG plus one TD3 change", fontweight="bold", fontsize=9, pad=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MUTED, label="no LayerNorm"),
               Line2D([], [], ls="", marker="o", ms=5, mfc=MUTED, mec=MUTED, label="critic LayerNorm")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="figs_extra")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    text = []
    main_groups = load_runs(["runs/core", "runs/beta075"])
    analysis_a(main_groups, text)
    analysis_b(summaries(main_groups), text)
    all_summ = summaries(load_runs(["runs/core", "runs/beta075", "runs/dose", "runs/ablation"]))
    sig_c = analysis_c(all_summ, a.out, text)
    sig_d = analysis_d(all_summ, a.out, text)
    if sig_c is not None and sig_d is not None:
        plot_controlled(all_summ, {**sig_c, **sig_d}, os.path.join(a.out, "controlled.png"))
    with open(os.path.join(a.out, "extra_stats.txt"), "w") as fh:
        fh.write("\n".join(text).translate(MINUS) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

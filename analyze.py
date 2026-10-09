"""Figures, statistics and the LaTeX results table from td3.py runs.

Usage: python analyze.py runs/core runs/beta075 [--out figs]
Only numpy / scipy / matplotlib, so it runs in the course's shared environment.

Conditions are (algorithm, beta, critic LayerNorm); runs without a "beta" in
their config are standard TD3/DDPG (beta = 1). Per run we define
  final return = mean evaluation return over the last FINAL_EVALS evaluations,
  bias         = mean bias over evaluations from BIAS_FROM x total steps onward,
  success      = share of the episodes of the last FINAL_EVALS evaluations whose
                 return is >= SUCCESS (secondary metric, added after the main results).
Curves are means over seeds with 95% percentile-bootstrap CIs, drawn only up to
the last step that every run of the group has reached (n in the legend).
A comparison is a linear contrast of condition means (a difference of two conditions,
or the LayerNorm x algorithm interaction) with a 95% bootstrap CI and a
Welch-Satterthwaite t-test; the table marks with a dagger the tests that remain
significant at 5% after a Holm correction over all the tests of the table.
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
from matplotlib.lines import Line2D
from scipy import stats

VARIANT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # DDPG, TD3, TD3 with beta < 1 (fixed order)
LN_TITLES = {0: "no LayerNorm", 1: "critic LayerNorm"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e4e0"
GUIDE = "#9a9a96"
PAGE_W = 6.9      # inches = the report's text width, so the fonts print at the size set here
MINUS = str.maketrans("-", "−")
WARMUP = 10_000   # first eval = end of random warm-up, critic barely trained
FINAL_EVALS = 3   # 280k, 290k, 300k
BIAS_FROM = 0.5   # bias averaged over the second half of training
SUCCESS = 200     # episode return that counts as a successful landing
# table columns: (index in the per-run summary, name, scale); success is shown in %
METRICS = ((0, "final", 1), (2, "success", 100), (1, "bias", 1))


def load(dirs):
    groups = defaultdict(list)  # (algo, beta, ln) -> list of runs
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*.json")):
            run = json.load(open(f))
            c = run["config"]
            groups[(c["algo"], c.get("beta", 1.0), c["critic_ln"])].append(run)
    return groups


def short_name(algo, beta):
    return "DDPG" if algo == "ddpg" else ("TD3" if beta == 1 else f"TD3, β = {beta:g}")


def table_name(algo, beta):
    return "DDPG" if algo == "ddpg" else ("TD3" if beta == 1 else rf"TD3, $\beta = {beta:g}$")


def variants(groups):
    """DDPG, then standard TD3, then the TD3 variants with a weighted target."""
    return sorted({(a, b) for a, b, _ in groups}, key=lambda k: (k[0] != "ddpg", k[1] != 1, k[1]))


def bootstrap_ci(x, n=10_000, seed=0):
    x = np.asarray(x)
    if len(x) < 2:
        return x.mean(), x.mean()
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def curve(runs, key, min_step=0):
    by_step = defaultdict(list)
    for run in runs:
        for e in run["evals"]:
            if e["step"] > min_step and key in e:
                by_step[e["step"]].append(e[key])
    # only steps reached by every run, so the mean is over a fixed set of seeds
    steps = np.array(sorted(s for s in by_step if len(by_step[s]) == len(runs)))
    mean = np.array([np.mean(by_step[s]) for s in steps])
    ci = np.array([bootstrap_ci(by_step[s]) for s in steps]).reshape(-1, 2)
    return steps, mean, ci


def summarize(run):
    """(final return, bias, success rate) of a finished run, or None if it is still running."""
    ev, total = run["evals"], run["config"]["total_steps"]
    if not ev or ev[-1]["step"] != total:
        return None
    final = np.mean([e["return_mean"] for e in ev[-FINAL_EVALS:]])
    bias = np.mean([e["bias"] for e in ev if e["step"] >= BIAS_FROM * total])
    success = np.mean([ret >= SUCCESS for e in ev[-FINAL_EVALS:] for ret in e["returns"]])
    return final, bias, success


def contrast(summ, terms, k, n_boot=10_000, seed=0):
    """sum(sign x condition mean) of metric k: estimate, 95% bootstrap CI, Welch-Satterthwaite p.

    With two conditions this is the difference of means and Welch's t-test."""
    xs = [np.array(summ[key])[:, k] for key, _ in terms]
    signs = [s for _, s in terms]
    est = sum(s * x.mean() for s, x in zip(signs, xs))
    rng = np.random.default_rng(seed)
    boot = sum(s * rng.choice(x, (n_boot, len(x))).mean(1) for s, x in zip(signs, xs))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    v = np.array([x.var(ddof=1) / len(x) for x in xs])
    se = np.sqrt(v.sum())
    df = v.sum() ** 2 / sum(vi ** 2 / (len(x) - 1) for vi, x in zip(v, xs))
    p = 2 * stats.t.sf(abs(est) / se, df) if se > 0 else 1.0
    return est, lo, hi, p


def holm(ps):
    """Holm-Bonferroni adjusted p-values, in the order of ps."""
    ps = np.asarray(ps, float)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(np.argsort(ps)):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def comparisons(summ, cols):
    """The comparisons of the table: (LaTeX label, plain label, [(condition, sign), ...])."""
    ok = lambda *keys: all(len(summ.get(k, [])) >= 2 for k in keys)
    out = []
    for ln in (0, 1):
        a, b = ("td3", 1.0, ln), ("ddpg", 1.0, ln)
        if ok(a, b):
            on = "on" if ln else "off"
            out.append((f"TD3 $-$ DDPG, LN {on}", f"TD3 - DDPG, LN {on}", [(a, 1), (b, -1)]))
    keys = [("td3", 1.0, 1), ("ddpg", 1.0, 1), ("td3", 1.0, 0), ("ddpg", 1.0, 0)]
    if ok(*keys):
        out.append((r"LN $\times$ algorithm", "interaction LN x algorithm", list(zip(keys, (1, -1, -1, 1)))))
    for algo, beta in cols:
        on, off = (algo, beta, 1), (algo, beta, 0)
        if ok(on, off):
            out.append((f"LN effect, {table_name(algo, beta)}", f"LN effect, {short_name(algo, beta)}",
                        [(on, 1), (off, -1)]))
    for algo, beta in cols:
        if beta == 1:
            continue
        for ln in (0, 1):
            a, b = (algo, beta, ln), (algo, 1.0, ln)
            if ok(a, b):
                on = "on" if ln else "off"
                out.append((rf"$\beta$ effect, LN {on}", f"beta effect ({beta:g} - 1), LN {on}", [(a, 1), (b, -1)]))
    return out


def signed(v):
    return f"{v:+.0f}".translate(MINUS)


def ptext(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def diff_heading(summ, a, b, k):
    """Panel heading: difference a - b of per-run metric k, with CI and Welch p."""
    if not all(len(summ.get(x, [])) >= 2 for x in (a, b)):
        return ""
    est, lo, hi, p = contrast(summ, [(a, 1), (b, -1)], k)
    return f"TD3 − DDPG: Δ = {signed(est)} [{signed(lo)}, {signed(hi)}], {ptext(p)}"


def plot_curves(groups, summ, cols, out):
    rows = [("return_mean", "Evaluation return", 0, 0), ("bias", "Bias  Q₁ − MC return", WARMUP, 1)]
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 4.9), sharex=True, sharey="row", squeeze=False)
    for i, (key, ylabel, min_step, k) in enumerate(rows):
        for j, ln in enumerate((0, 1)):
            ax = axes[i, j]
            ax.axhline(0, color=GUIDE, lw=0.6)
            if key == "return_mean":
                ax.axhline(200, color=GUIDE, lw=0.6, ls=":")
            for c, (algo, beta) in enumerate(cols):
                runs = groups.get((algo, beta, ln))
                if not runs:
                    continue
                steps, mean, ci = curve(runs, key, min_step)
                ax.plot(steps / 1e3, mean, color=VARIANT_COLORS[c], lw=1.6,
                        label=f"{short_name(algo, beta)} (n = {len(runs)} seeds)")
                ax.fill_between(steps / 1e3, ci[:, 0], ci[:, 1], color=VARIANT_COLORS[c], alpha=0.18, lw=0)
            heading = diff_heading(summ, ("td3", 1.0, ln), ("ddpg", 1.0, ln), k)
            if i == 0:
                ax.set_title(LN_TITLES[ln], fontweight="bold", fontsize=9.5, pad=14)
                ax.text(0.5, 1.015, heading, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
            else:
                ax.set_title(heading, fontsize=7.5, color=MUTED, pad=4)
            ax.set_ylabel(ylabel)                   # every panel labelled and scaled
            ax.set_xlabel("Environment steps (×1000)")
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.grid(axis="y", color=GRID, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(cols), frameon=False)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_scatter(summ, cols, out):
    fig, axes = plt.subplots(1, len(cols), figsize=(PAGE_W, 2.8), sharey=True, squeeze=False)
    for c, (algo, beta) in enumerate(cols):
        ax, color = axes[0, c], VARIANT_COLORS[c]
        pts = []
        for ln in (0, 1):  # hollow: no LayerNorm, filled: critic LayerNorm
            s = summ.get((algo, beta, ln), [])
            if s:
                f, b = np.array(s)[:, :2].T
                ax.scatter(b, f, s=26, facecolor=color if ln else "white", edgecolor=color, lw=1.2, zorder=3)
                pts += s
        if len(pts) >= 3:
            f, b = np.array(pts)[:, :2].T
            rho, p = stats.spearmanr(b, f)
            ax.text(0.5, 1.015, f"Spearman ρ = {rho:+.2f}, {ptext(p)}".translate(MINUS),
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=MUTED)
        ax.axhline(200, color=GUIDE, lw=0.6, ls=":")
        ax.set_title(short_name(algo, beta), fontweight="bold", fontsize=9.5, pad=14)
        ax.set_xlabel(f"Bias ({BIAS_FROM:.0%}–100% of training)")
        ax.set_ylabel(f"Final return (last {FINAL_EVALS} evals)")
        ax.tick_params(labelleft=True)
        ax.grid(color=GRID, lw=0.5)
    handles = [Line2D([], [], ls="", marker="o", ms=5.5, mfc="white", mec=MUTED, mew=1.2, label="no LayerNorm"),
               Line2D([], [], ls="", marker="o", ms=5.5, mfc=MUTED, mec=MUTED, label="critic LayerNorm")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fmt(m, lo, hi, scale=1):
    return f"${scale * m:.0f}$ {{\\scriptsize[${scale * lo:.0f}$, ${scale * hi:.0f}$]}}"


def fmt_p(p):
    return "$p<0.001$" if p < 0.001 else f"$p={p:.2g}$"


def results_table(summ, cols, out):
    """LaTeX tabular: per-condition means with 95% CIs, then the comparisons."""
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"Algorithm & LN & $n$ & Final return & Success (\%) & Bias \\", r"\midrule"]
    text = []
    for algo, beta in cols:
        for ln in (0, 1):
            s = summ.get((algo, beta, ln), [])
            if not s:
                continue
            arr = np.array(s)
            stat = [(metric, scale, arr[:, k].mean(), *bootstrap_ci(arr[:, k])) for k, metric, scale in METRICS]
            lines.append(f"{table_name(algo, beta)} & {'on' if ln else 'off'} & {len(s)} & "
                         + " & ".join(fmt(m, lo, hi, scale) for _, scale, m, lo, hi in stat) + r" \\")
            text.append(f"{short_name(algo, beta):14s} LN={ln} n={len(s):2d}  " + "  ".join(
                f"{metric} {scale * m:6.1f} [{scale * lo:6.1f}, {scale * hi:6.1f}]" for metric, scale, m, lo, hi in stat))
    specs = comparisons(summ, cols)
    results = [[contrast(summ, terms, k) for k, _, _ in METRICS] for _, _, terms in specs]
    adj = holm([r[3] for row in results for r in row]).reshape(len(specs), len(METRICS))
    rows = []
    for (label_tex, label_txt, _), row, row_adj in zip(specs, results, adj):
        cells = []
        for (est, lo, hi, p), pa, (_, metric, scale) in zip(row, row_adj, METRICS):
            cells.append(f"{fmt(est, lo, hi, scale)}, {fmt_p(p)}" + (r"$^\dagger$" if pa < 0.05 else ""))
            text.append(f"{label_txt:32s} {metric:7s}: {scale * est:+7.1f} [{scale * lo:7.1f}, {scale * hi:7.1f}]"
                        f"  p={p:.3g}  Holm p={pa:.3g}")
        rows.append(f"\\multicolumn{{3}}{{@{{}}l}}{{{label_tex}}} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    text.append(f"tests in the table: {adj.size} (dagger = significant after Holm correction)")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default="figs")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})

    groups = load(a.dirs)
    cols = variants(groups)
    summ, unfinished = defaultdict(list), 0
    for k, runs in groups.items():
        for run in runs:
            s = summarize(run)
            if s is None:
                unfinished += 1
            else:
                summ[k].append(s)

    plot_curves(groups, summ, cols, os.path.join(a.out, "curves.png"))
    plot_scatter(summ, cols, os.path.join(a.out, "scatter.png"))
    text = results_table(summ, cols, os.path.join(a.out, "results_table.tex"))
    allpts = [x for s in summ.values() for x in s]
    if len(allpts) >= 3:
        f, b = np.array(allpts)[:, :2].T
        rho, pv = stats.spearmanr(b, f)
        text.append(f"pooled over all finished runs: Spearman rho={rho:+.3f} p={pv:.3g} n={len(allpts)}")
    text.append(f"finished runs: {len(allpts)}, still running: {unfinished}")
    with open(os.path.join(a.out, "stats.txt"), "w") as fh:
        fh.write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

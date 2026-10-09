"""Figures and statistics for the project. Reads results/ only (no torch needed).

    python analysis.py                        # "core" experiment, plus "bias" if it exists
    python analysis.py --bias-metric bias_norm --last 5 --out figures

Writes to figures/ (each figure as .pdf and .png):
    curves             learning curves (a) and bias curves (b), mean and 95% CI over seeds
    bias_vs_return     final bias against final return, one point per run
    bias_dial          only if results/bias exists: effect of the number of critics (TD3)
    stats_contrasts.csv, stats_correlations.csv, (stats_dial.csv)
and prints the number of seeds per configuration (put it in the figure captions).

Bias = Q(s,a) - Monte Carlo return of the same (s,a), averaged over evaluation states ("bias") or divided by
the mean |return| ("bias_norm"; unstable if the mean return is close to 0, hence "bias" is the default).
Statistics use ONE number per run (its final value: mean of the last evaluations), never evaluation episodes.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .experiment import Experiment, load_many

# (algo, critic_ln) -> label, colour (Okabe-Ito, colour-blind safe), line style, marker
STYLE = {
    ("ddpg", False): ("DDPG", "#0072B2", "-", "o"),
    ("ddpg", True): ("DDPG + LayerNorm", "#56B4E9", "--", "s"),
    ("td3", False): ("TD3", "#D55E00", "-", "^"),
    ("td3", True): ("TD3 + LayerNorm", "#E69F00", "--", "D"),
}
BIAS_LABEL = {"bias": "Bias: Q $-$ Monte Carlo return", "bias_norm": "Normalised bias: (Q $-$ MC) / |MC|"}
RETURN_LABEL = "Evaluation return"
# contrasts (name, reference config, compared config); the difference is always compared - reference
CONTRASTS = [
    ("LayerNorm effect, DDPG", ("ddpg", False), ("ddpg", True)),
    ("LayerNorm effect, TD3", ("td3", False), ("td3", True)),
    ("TD3 - DDPG, without LayerNorm", ("ddpg", False), ("td3", False)),
    ("TD3 - DDPG, with LayerNorm", ("ddpg", True), ("td3", True)),
]

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def select(df, algo, ln):
    return df[(df.algo == algo) & (df.critic_ln == ln)]


def t_ci(values):
    """Half-width of the 95% t confidence interval of the mean."""
    n = len(values)
    return stats.t.ppf(0.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n) if n > 1 else np.nan


# ---- figures -----------------------------------------------------------------------------------
def plot_curves(evals, bias_metric, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for (algo, ln), (label, color, ls, _) in STYLE.items():
        g = select(evals, algo, ln).groupby("step")
        for ax, metric in zip(axes, ("return_mean", bias_metric)):
            mean, n = g[metric].mean(), g[metric].count()
            ci = stats.t.ppf(0.975, (n - 1).clip(lower=1)) * g[metric].std() / np.sqrt(n)
            x = mean.index.values / 1000
            ax.plot(x, mean.values, color=color, ls=ls, lw=1.5, label=label)
            ax.fill_between(x, (mean - ci).values, (mean + ci).values, color=color, alpha=0.2, lw=0)
    axes[0].set_ylabel(RETURN_LABEL)
    axes[1].set_ylabel(BIAS_LABEL[bias_metric])
    axes[1].axhline(0, color="k", lw=0.6)
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("Environment steps ($\\times 10^3$)")
        ax.text(-0.02, 1.05, f"({letter})", transform=ax.transAxes, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "curves")


def plot_scatter(summary, bias_metric, out):
    col = f"final_{bias_metric}"
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for (algo, ln), (label, color, _, marker) in STYLE.items():
        d = select(summary, algo, ln)
        ax.scatter(d[col], d.final_return_mean, color=color, marker=marker, s=30, edgecolor="k", linewidth=0.4,
                   label=label)
    rho, p = stats.spearmanr(summary[col], summary.final_return_mean)
    ax.text(0.03, 0.04, f"Spearman $\\rho$ = {rho:.2f} (n = {len(summary)})", transform=ax.transAxes, fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Final " + BIAS_LABEL[bias_metric][0].lower() + BIAS_LABEL[bias_metric][1:])
    ax.set_ylabel("Final " + RETURN_LABEL.lower())
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, out, "bias_vs_return")


def plot_dial(dial, bias_metric, out):
    """TD3 only: final bias and final return against the number of critics in the clipped-min target."""
    col = f"final_{bias_metric}"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ln, off in ((False, -0.05), (True, 0.05)):
        _, color, ls, marker = STYLE[("td3", ln)]
        label = "TD3 + LayerNorm" if ln else "TD3"
        d = select(dial, "td3", ln)
        for ax, metric in zip(axes, (col, "final_return_mean")):
            ax.scatter(d.n_critics + off, d[metric], color=color, marker=marker, s=22, alpha=0.7, label=label)
            g = d.groupby("n_critics")[metric]
            ax.errorbar(g.mean().index + off, g.mean().values, yerr=g.agg(t_ci).values, color=color, ls=ls,
                        lw=1.5, capsize=3)
    axes[0].set_ylabel("Final " + BIAS_LABEL[bias_metric][0].lower() + BIAS_LABEL[bias_metric][1:])
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("Final " + RETURN_LABEL.lower())
    for ax, letter in zip(axes, "ab"):
        ax.set_xlabel("Number of critics in the min target")
        ax.set_xticks(sorted(dial.n_critics.unique()))
        ax.text(-0.02, 1.05, f"({letter})", transform=ax.transAxes, fontweight="bold")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "bias_dial")


# ---- statistics --------------------------------------------------------------------------------
def welch(a, b):
    """Difference of means b - a with its 95% Welch CI, Hedges' g, and the Welch and Mann-Whitney p-values."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    diff, se = np.mean(b) - np.mean(a), np.sqrt(va / na + vb / nb)
    df = se**4 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    half = stats.t.ppf(0.975, df) * se
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    g = diff / pooled * (1 - 3 / (4 * (na + nb) - 9))
    return dict(mean_ref=np.mean(a), mean_cmp=np.mean(b), diff=diff, ci_low=diff - half, ci_high=diff + half,
                hedges_g=g, p_welch=stats.ttest_ind(b, a, equal_var=False).pvalue,
                p_mannwhitney=stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def contrasts(summary, bias_metric):
    rows = []
    for metric in ("final_return_mean", f"final_{bias_metric}"):
        block = []
        for name, ref, cmp in CONTRASTS:
            a = select(summary, *ref)[metric].dropna().values
            b = select(summary, *cmp)[metric].dropna().values
            block.append(dict(metric=metric, contrast=name, n_ref=len(a), n_cmp=len(b), **welch(a, b)))
        for row, p in zip(block, holm([r["p_welch"] for r in block])):
            row["p_welch_holm"] = p
        rows += block
    return pd.DataFrame(rows)


def correlations(summary, bias_metric):
    """Spearman correlation between final bias and final return: pooled, then within each configuration
    (pooling mixes the effect of the configuration with the effect of the bias)."""
    col, rows = f"final_{bias_metric}", []
    groups = [("all runs", summary)] + [(STYLE[k][0], select(summary, *k)) for k in STYLE]
    for name, d in groups:
        rho, p = stats.spearmanr(d[col], d.final_return_mean) if len(d) > 2 else (np.nan, np.nan)
        rows.append(dict(group=name, n=len(d), spearman_rho=rho, p=p))
    return pd.DataFrame(rows)


# ---- main --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--bias-metric", default="bias", choices=["bias", "bias_norm"])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.float_format", "{:.3g}".format)

    core = Experiment.from_dir(Path(a.root) / "core")
    evals, summary = core.load("evals"), core.summary()
    print("finished runs (seeds) per configuration:")
    print(summary.groupby("config_id").size().to_string(), "\n")
    plot_curves(evals, a.bias_metric, out)
    plot_scatter(summary, a.bias_metric, out)

    table = contrasts(summary, a.bias_metric)
    table.to_csv(out / "stats_contrasts.csv", index=False)
    corr = correlations(summary, a.bias_metric)
    corr.to_csv(out / "stats_correlations.csv", index=False)
    print(table.to_string(index=False), "\n")
    print(corr.to_string(index=False), "\n")

    if (Path(a.root) / "bias" / "experiment.json").exists():
        both = load_many(["core", "bias"], root=a.root)
        dial = both[both.algo == "td3"]
        print("bias experiment: runs per (LayerNorm, n_critics):")
        print(dial.groupby(["critic_ln", "n_critics"]).size().to_string(), "\n")
        plot_dial(dial, a.bias_metric, out)
        rows = []
        for name, d in [("TD3 all", dial)] + [(f"TD3 LayerNorm={ln}", select(dial, "td3", ln)) for ln in (False, True)]:
            for x, y in ((f"final_{a.bias_metric}", "final_return_mean"), ("n_critics", f"final_{a.bias_metric}"),
                         ("n_critics", "final_return_mean")):
                rho, pv = stats.spearmanr(d[x], d[y])
                rows.append(dict(group=name, x=x, y=y, n=len(d), spearman_rho=rho, p=pv))
        pd.DataFrame(rows).to_csv(out / "stats_dial.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nfigures and tables written to {out}/")


if __name__ == "__main__":
    main()

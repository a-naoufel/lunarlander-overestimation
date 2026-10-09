"""Entry point: runs the experiments of the project.

    python run_all.py --smoke                    # 1. tiny sanity run of the core grid (a few minutes)
    python run_all.py core --workers 4           # 2. the must-have experiment: 4 configs x 5 seeds
    python run_all.py bias --workers 4           # 3. controlled bias manipulation on TD3
    python run_all.py depth                      # 4. optional extras (pick at most one in a short project)
    python run_all.py all --workers 4            # everything

Finished runs are skipped, so an interrupted command can simply be re-run, and extra seeds can be added
later (--seeds 0-9). Results: results/<experiment>/<config_id>/seed<k>/ and results/<experiment>/experiment.json.
Watch training with:  tensorboard --logdir results
Analysis (no torch needed):
    from experiment import Experiment, load_many
    Experiment.from_dir("results/core").summary()       # one row per run
    load_many(["core", "bias"], what="evals")           # learning / bias curves of several experiments
"""
import argparse

from .experiment import Experiment, parse_seeds

# Overrides of the Config defaults (config.py) applied to every experiment. The defaults are the
# project's shared setting: Stable-Baselines3 DDPG/TD3, 256x256 networks, lr 1e-3, gamma 0.98, tau 0.01,
# batch 256, 150k steps, buffer 100k, 5k random warm-up steps, exploration noise 0.1.
BASE = {}

# Each experiment is a grid: every combination of the listed values is one configuration, run on every seed.
# The configurations of different experiments do not overlap, so nothing is trained twice.
EXPERIMENTS = {
    # Q1 + Q2. Does LayerNorm change the bias, for DDPG and for TD3? Does the bias relate to performance?
    "core": dict(grid=dict(algo=["ddpg", "td3"], critic_ln=[False, True])),
    # Q2, causally. Instead of only observing bias, push it: the TD3 target is the min over n_critics target
    # critics. 1 critic = no clipped double-Q (most overestimation), 5 = more pessimistic than standard.
    # (n_critics=2 is standard TD3, already in "core".)
    "bias": dict(grid=dict(algo=["td3"], critic_ln=[False, True], n_critics=[1, 5])),
    # Optional extras: one more factor each, the other value being the "core" setting.
    "depth": dict(grid=dict(algo=["ddpg", "td3"], critic_ln=[False, True], n_layers=[3])),
    "utd": dict(grid=dict(algo=["ddpg", "td3"], critic_ln=[False, True], utd=[4])),  # 4x compute per run
}

SMOKE = dict(total_steps=3_000, learning_starts=500, buffer_size=3_000, eval_every=1_000,
             eval_episodes=2, log_every=500)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="*", metavar="EXP",
                   help=f"one or more of {list(EXPERIMENTS)} or 'all' (default: core)")
    p.add_argument("--seeds", nargs="+", default=["0-4"], help="e.g. 0-4 or 0 1 2")
    p.add_argument("--workers", type=int, default=4, help="runs in parallel (about one per CPU core)")
    p.add_argument("--root", default="results")
    p.add_argument("--smoke", action="store_true", help="tiny runs in results_smoke/ to check that everything works")
    a = p.parse_args()

    names = a.experiments or ["core"]
    if "all" in names:
        names = list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiment(s) {unknown}; choose from {list(EXPERIMENTS)} or 'all'")
    seeds = parse_seeds(a.seeds)
    root = a.root
    if a.smoke:
        seeds, root = seeds[:2], "results_smoke"
    for name in names:
        spec = EXPERIMENTS[name]
        base = {**BASE, **spec.get("base", {}), **(SMOKE if a.smoke else {})}
        Experiment(name, base, spec["grid"], seeds, root).run(a.workers)


if __name__ == "__main__":
    main()

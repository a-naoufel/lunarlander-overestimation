"""An Experiment = base settings + a grid of varied hyper-parameters + a list of seeds.

    results/<name>/
        experiment.json                 spec: base, grid, seeds and the full config of every grid point
        <config_id>/seed<k>/            one run (see logger.py), e.g. "algo=td3,critic_ln=1/seed3"

Command line (re-running the same command skips finished runs, so you can add seeds later):

    python experiment.py core --grid algo=ddpg,td3 critic_ln=0,1 --seeds 0-4 --workers 5

Python (e.g. in an analysis notebook, no torch needed for this part):

    exp = Experiment.from_dir("results/core")
    exp.load("evals")      # every evaluation of every run, with config_id, seed and the varied params
    exp.summary()          # one row per run: final return / bias averaged over the last evaluations
"""
import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from multiprocessing import get_context
from pathlib import Path
from typing import get_args

import pandas as pd

from .config import Config


def config_id(varied):
    if not varied:
        return "base"
    return ",".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in varied.items())


def _run_one(cfg_dict, run_dir):
    from .trainer import Trainer  # imported here: workers import torch themselves, analysis code never does
    Trainer(Config.from_dict(cfg_dict), run_dir).run()


class Experiment:
    def __init__(self, name, base=None, grid=None, seeds=(0,), root="results"):
        self.name = name
        self.base = dict(base or {})
        self.grid = {k: list(v) for k, v in (grid or {}).items()}
        self.seeds = list(seeds)
        self.dir = Path(root) / name

    # ---- spec ----------------------------------------------------------------------------------
    def configs(self):
        """{config_id: varied params} for every point of the grid."""
        out = {}
        for values in itertools.product(*self.grid.values()):
            varied = dict(zip(self.grid, values))
            out[config_id(varied)] = varied
        return out

    def _index(self):
        for cid, varied in self.configs().items():
            for seed in self.seeds:
                yield cid, varied, seed, self.dir / cid / f"seed{seed}"

    def runs(self):
        """(config_id, seed, Config, run_dir) for every run."""
        for cid, varied, seed, run_dir in self._index():
            yield cid, seed, Config(**{**self.base, **varied, "seed": seed}), run_dir

    def save_spec(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        full = {}
        for cid, varied in self.configs().items():
            d = Config(**{**self.base, **varied}).to_dict()
            d.pop("seed")
            full[cid] = d
        spec = {"name": self.name, "base": self.base, "grid": self.grid, "seeds": self.seeds, "configs": full}
        (self.dir / "experiment.json").write_text(json.dumps(spec, indent=2))

    @classmethod
    def from_dir(cls, path):
        path = Path(path)
        spec = json.loads((path / "experiment.json").read_text())
        return cls(path.name, spec["base"], spec["grid"], spec["seeds"], root=path.parent)

    # ---- running -------------------------------------------------------------------------------
    def run(self, workers=1):
        self.save_spec()
        runs = list(self.runs())
        todo = [(cfg.to_dict(), str(d)) for _, _, cfg, d in runs if not (d / "summary.json").exists()]
        print(f"{self.dir}: {len(todo)} runs to do, {len(runs) - len(todo)} already finished")
        if not todo:
            return
        if workers <= 1:
            for cfg_dict, run_dir in todo:
                _run_one(cfg_dict, run_dir)
        else:
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
                list(pool.map(_run_one, *zip(*todo)))  # list() re-raises errors from the workers
        print(f"done. TensorBoard: tensorboard --logdir {self.dir}")

    # ---- results -------------------------------------------------------------------------------
    def load(self, table="evals"):
        """One of 'evals', 'internals', 'episodes' for all runs, with config_id, seed and varied params."""
        frames = []
        for cid, varied, seed, run_dir in self._index():
            f = run_dir / f"{table}.csv"
            if f.exists():
                df = pd.read_csv(f)
                for k, v in reversed(list(varied.items())):
                    df.insert(0, k, v)
                df.insert(0, "seed", seed)
                df.insert(0, "config_id", cid)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def summary(self):
        """One row per finished run: final_return_mean, final_bias, ... (mean of the last evaluations)."""
        rows = []
        for cid, varied, seed, run_dir in self._index():
            f = run_dir / "summary.json"
            if f.exists():
                rows.append({"config_id": cid, "seed": seed, **varied, **json.loads(f.read_text())})
        return pd.DataFrame(rows)

    def raw_eval(self, cid, seed, step):
        """Raw per-state arrays (q, g, episode, t, returns) of one evaluation of one run."""
        import numpy as np
        return dict(np.load(self.dir / cid / f"seed{seed}" / "eval_raw" / f"step_{step:08d}.npz"))


def load_many(names, root="results", what="summary"):
    """summary() or load(table) of several experiments stacked, with an 'experiment' column. Parameters
    varied in some experiments are filled in for the others from their saved full configs."""
    exps = [Experiment.from_dir(Path(root) / n) for n in names]
    keys = list(dict.fromkeys(k for e in exps for k in e.grid))
    frames = []
    for e in exps:
        df = e.summary() if what == "summary" else e.load(what)
        if df.empty:
            continue
        full = json.loads((e.dir / "experiment.json").read_text())["configs"]
        for k in keys:
            if k not in df:
                df[k] = df["config_id"].map(lambda cid: full[cid][k])
        df.insert(0, "experiment", e.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---- command line ------------------------------------------------------------------------------
def _cast(key, text):
    types = {f.name: f.type for f in fields(Config)}
    if key not in types:
        raise SystemExit(f"unknown hyper-parameter {key!r}")
    t = types[key]
    if get_args(t):  # Optional[int]
        t = get_args(t)[0]
    if t is bool:
        return text.lower() in ("1", "true", "yes")
    return t(text)


def _parse_pairs(items, multi):
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        values = [_cast(key, v) for v in value.split(",")]
        out[key] = values if multi else values[0]
    return out


def parse_seeds(tokens):
    seeds = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-")
            seeds += range(int(lo), int(hi) + 1)
        else:
            seeds.append(int(tok))
    return seeds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="experiment name (results go to <root>/<name>)")
    p.add_argument("--grid", nargs="*", default=[], metavar="KEY=V1,V2", help="hyper-parameters to vary")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=V", help="fixed hyper-parameters")
    p.add_argument("--seeds", nargs="+", default=["0"], help="e.g. 0 1 2 or 0-4")
    p.add_argument("--workers", type=int, default=1, help="runs in parallel")
    p.add_argument("--root", default="results")
    a = p.parse_args()
    Experiment(a.name, _parse_pairs(a.set, multi=False), _parse_pairs(a.grid, multi=True),
               parse_seeds(a.seeds), a.root).run(a.workers)


if __name__ == "__main__":
    main()

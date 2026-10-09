"""Writes everything a run produces into its directory:

    config.json      hyper-parameters
    internals.csv    networks' internals (losses, Q-values, gradient/weight norms, ...) every log_every steps
    episodes.csv     one row per training episode: step, return, length
    evals.csv        one row per evaluation: return and Q-vs-Monte-Carlo bias metrics
    eval_raw/        raw (Q, MC return, episode, t) of every visited state, one .npz per evaluation
    summary.json     final numbers of the run (its presence marks the run as finished)
    events.*         TensorBoard logs (same scalars)
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


class RunLogger:
    SUMMARY_KEYS = ("return_mean", "bias", "bias_norm", "bias_min", "q_gap", "q_mean", "mc_mean")

    def __init__(self, run_dir, cfg):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        (self.dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        self.rows = {"internals": [], "episodes": [], "evals": []}
        self.tb = None
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.dir))
            except ImportError:
                print("tensorboard is not installed: only csv/json logs will be written")

    def _add(self, table, tag, step, values):
        self.rows[table].append({"step": step, **values})
        if self.tb:
            for k, v in values.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.tb.add_scalar(f"{tag}/{k}", v, step)

    def log_internals(self, step, stats):
        self._add("internals", "internals", step, stats)

    def log_episode(self, step, ret, length):
        self._add("episodes", "train", step, {"return": float(ret), "length": int(length)})

    def log_eval(self, step, metrics, raw=None):
        self._add("evals", "eval", step, metrics)
        if raw and self.cfg.save_raw:
            (self.dir / "eval_raw").mkdir(exist_ok=True)
            np.savez_compressed(self.dir / "eval_raw" / f"step_{step:08d}.npz", **raw)
        self.flush()

    def flush(self):
        for name, rows in self.rows.items():
            if rows:
                pd.DataFrame(rows).to_csv(self.dir / f"{name}.csv", index=False)
        if self.tb:
            self.tb.flush()

    def finish(self, extra=None):
        self.flush()
        evals = pd.DataFrame(self.rows["evals"]).tail(self.cfg.final_window)
        summary = {"status": "done", "n_evals": len(self.rows["evals"]), **(extra or {})}
        for key in self.SUMMARY_KEYS:
            if key in evals:
                summary[f"final_{key}"] = float(evals[key].mean())
        (self.dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
        if self.tb:
            self.tb.close()

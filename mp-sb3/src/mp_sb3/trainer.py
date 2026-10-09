import time
from pathlib import Path

import numpy as np
import torch 
from stable_baselines3.common.callbacks import BaseCallback

from .evaluation import Evaluator
from mp_sb3.logger import RunLogger
from mp_sb3.models import SB3Agent, build_model, make_env


class _StepHook(BaseCallback):
    """SB3 calls this after every environment step; it just forwards to the Trainer."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def _on_step(self):
        self.trainer.after_step(self.num_timesteps, self.locals["infos"])
        return True


def _l2(tensors):
    return torch.sqrt(sum(t.detach().pow(2).sum() for t in tensors)).item()


class Trainer:
    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        torch.set_num_threads(cfg.threads)
        self.model = build_model(cfg, make_env(cfg))
        self.agent = SB3Agent(self.model)
        self.evaluator = Evaluator(cfg, make_env(cfg))
        self.logger = RunLogger(run_dir, cfg)
        self.label = "/".join(Path(run_dir).parts[-2:])
        self.t0 = time.time()

    def run(self):
        self.t0 = time.time()
        self.model.learn(total_timesteps=self.cfg.total_steps, callback=_StepHook(self))
        torch.save(self._weights(), self.logger.dir / "final.pt")
        self.logger.finish({"steps": self.cfg.total_steps, "time_s": time.time() - self.t0})

    def after_step(self, step, infos):
        c = self.cfg
        episode = infos[0].get("episode")
        if episode is not None:
            self.logger.log_episode(step, episode["r"], episode["l"])
        if step > c.learning_starts and step % c.log_every == 0:
            self.logger.log_internals(step, self._internals())
        if step % c.eval_every == 0:
            self._evaluate(step)

    def _weights(self):
        return {"actor": self.model.actor.state_dict(), "critic": self.model.critic.state_dict()}

    def _internals(self):
        m = self.model
        stats = {}
        recorded = getattr(m.logger, "name_to_value", {})  # losses SB3 recorded at its last train() call
        for key in ("critic_loss", "actor_loss"):
            if f"train/{key}" in recorded:
                stats[key] = float(recorded[f"train/{key}"])

        rng_state = np.random.get_state()
        batch = m.replay_buffer.sample(self.cfg.batch_size)
        np.random.set_state(rng_state)
        with torch.no_grad():
            q = torch.cat(m.critic(batch.observations, batch.actions), dim=1)
            a = m.actor(batch.observations)
        for i in range(q.shape[1]):
            stats[f"q{i + 1}_mean"] = q[:, i].mean().item()
        if q.shape[1] > 1:
            stats["q_gap"] = (q.max(1).values - q.min(1).values).mean().item()
        stats["action_abs_mean"] = a.abs().mean().item()
        stats["action_saturation"] = (a.abs() > 0.99).float().mean().item()
        stats["actor_weight_norm"] = _l2(m.actor.parameters())
        stats["critic_weight_norm"] = _l2(m.critic.parameters())
        grads = [p.grad for p in m.actor.parameters() if p.grad is not None]
        if grads:
            stats["actor_grad_norm"] = _l2(grads)
        return stats

    def _evaluate(self, step):
        elapsed = time.time() - self.t0
        metrics, raw = self.evaluator.run(self.agent)
        metrics["time"] = elapsed
        self.logger.log_eval(step, metrics, raw)
        if self.cfg.save_ckpt:
            (self.logger.dir / "ckpts").mkdir(exist_ok=True)
            torch.save(self._weights(), self.logger.dir / "ckpts" / f"step_{step:08d}.pt")

            
        print(f"[{self.label}] step {step}  return {metrics['return_mean']:7.1f}  "
              f"bias {metrics.get('bias', float('nan')):7.2f}  Q {metrics.get('q_mean', float('nan')):7.1f}  "
              f"MC {metrics.get('mc_mean', float('nan')):7.1f}  {elapsed:.0f}s", flush=True)

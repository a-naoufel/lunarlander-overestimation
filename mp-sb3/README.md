# mp-sb3: replication with Stable-Baselines3

DDPG and TD3 from [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on
`LunarLander-v3` (continuous), with optional LayerNorm in the critics and a Monte Carlo estimate of
the critics' overestimation bias. Part of the mini-project in the parent directory.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build (uv.lock pins the CUDA one)
uv pip install -e .
python -m mp_sb3.run_all --smoke                   # tiny runs to check the setup
python -m mp_sb3.run_all core bias --workers 20    # 4 + 4 configurations x seeds 0-4
python -m mp_sb3.analysis                          # figures/ and figures/stats_*.csv
# the same grid with the 300k-step budget of the main runs (results_300k/):
python -m mp_sb3.experiment core --grid algo=ddpg,td3 critic_ln=0,1 --set total_steps=300000 --seeds 0-4 --workers 20 --root results_300k
```

Settings are in `src/mp_sb3/config.py`: 150k steps, γ = 0.98, learning rate 1e-3, τ = 0.01,
100k buffer, 5k random warm-up steps, evaluation every 5k steps (10 deterministic episodes).
The runs of the report used Python 3.12, Stable-Baselines3 2.9.0, PyTorch 2.14.1 (CPU) and
Gymnasium 1.4.0.

Each run is in `results/<experiment>/<configuration>/seed<k>/`: `config.json`, `evals.csv` (every
evaluation: return and Q-vs-Monte-Carlo bias), `eval_returns.csv` (return of every evaluation
episode), `episodes.csv` (training episodes), `internals.csv` (losses, Q-values, norms) and
`summary.json` (final values, mean of the last 5 evaluations). The raw per-state arrays
(`eval_raw/`), TensorBoard logs and final weights are not in the repository.

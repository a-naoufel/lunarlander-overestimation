#!/usr/bin/env bash
# Usage: [ALGOS="ddpg td3"] ./launch.sh <out_dir> <n_parallel> [extra td3.py args...]
# Runs $ALGOS x critic LayerNorm {0,1} x seeds 0..9, one log per run in <out_dir>/logs.
# Core grid:   ./launch.sh runs/core 18
# Beta sweep:  ALGOS=td3 ./launch.sh runs/beta075 20 --beta 0.75 --save-ckpt 1
OUT=$1; P=$2; shift 2
ALGOS=${ALGOS:-"ddpg td3"}
mkdir -p "$OUT/logs"
for seed in $(seq 0 9); do for algo in $ALGOS; do for ln in 0 1; do
  echo "$algo $ln $seed"
done; done; done | xargs -P "$P" -L1 sh -c \
  '.venv/bin/python td3.py --algo $0 --critic-ln $1 --seed $2 --out "'"$OUT"'" '"$*"' > "'"$OUT"'/logs/${0}_ln${1}_s${2}.log" 2>&1'

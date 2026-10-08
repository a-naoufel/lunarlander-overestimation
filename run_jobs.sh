#!/usr/bin/env bash
# Usage: ./run_jobs.sh <jobs file> <n_parallel>
# Each line of the jobs file holds the td3.py arguments of one run; one log per run in runs/logs/.
JOBS=$1; P=$2
mkdir -p runs/logs
xargs -P "$P" -I{} sh -c '.venv/bin/python td3.py {} > "runs/logs/$(echo "{}" | tr -s " /." "___" | tr -d "-").log" 2>&1' < "$JOBS"

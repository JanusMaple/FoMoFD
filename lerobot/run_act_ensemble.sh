#!/usr/bin/env bash
# Use directly on a desktop or via sbatch on PACE (one GPU per process).
# Supply cluster-specific account/partition/resources to sbatch, not in this file.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_act_ensemble.sh TASK SEED [runner options]" >&2
  echo "For Slurm, export FOMOFD_REPO and FOMOFD_PYTHON before sbatch." >&2
  exit 2
fi
task=$1
seed=$2
shift 2
case "$task" in
  needle_pickup|ring_over_post|tissue_retraction|shunt_insertion) ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac
case "$seed" in 1001|1002) ;; *) echo "Expected seed 1001 or 1002" >&2; exit 2 ;; esac

if [[ -n "${SLURM_JOB_ID:-}" && -z "${FOMOFD_REPO:-}" ]]; then
  echo "Set FOMOFD_REPO: Slurm copies the script away from the checkout." >&2
  exit 2
fi
repo=${FOMOFD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python_bin=${FOMOFD_PYTHON:-python}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
cd "$repo"
exec "$python_bin" -u lerobot/train_act_ensemble.py train \
  --config "lerobot/configs/$task/policy_ensemble/seed$seed.json" "$@"

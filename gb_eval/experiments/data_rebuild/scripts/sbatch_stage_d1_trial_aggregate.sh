#!/bin/bash
#SBATCH --job-name=data_d1_trial_agg
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/stage_d1_trial_agg_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_d1_trial_agg_%A_%a.err   # edit for your cluster (relative to the submit directory)
# Stage D1-4 aggregation -> stage_d1_match_trial.json
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
# The results directory is switched with the environment variable DATA_RESULTS_DIR (if unset, the production RUN/results).
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} results=${DATA_RESULTS_DIR:-default}"
"$PY" stage_d1_match_trial.py aggregate

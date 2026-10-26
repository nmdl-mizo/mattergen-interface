#!/bin/bash
#SBATCH --job-name=data_stage_b
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/stage_b_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_b_%j.err   # edit for your cluster (relative to the submit directory)
# Data rebuild, stage B (1 to 5). Submit after the stage A check passed. --dependency=afterok:<stage A> is enough
# The target templates are narrowed with the environment variable DATA_CNIDS (comma separated). If unset, all 31 templates.
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
echo "host=$(hostname) job=${SLURM_JOB_ID} cnids=${DATA_CNIDS:-ALL}"
"$PY" stage_b_removal.py

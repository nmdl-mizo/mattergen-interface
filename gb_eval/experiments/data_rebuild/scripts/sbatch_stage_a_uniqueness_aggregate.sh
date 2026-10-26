#!/bin/bash
#SBATCH --job-name=data_uniq_agg
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/uniq_agg_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/uniq_agg_%j.err   # edit for your cluster (relative to the submit directory)
# Data rebuild, stage A-5 aggregation. Submission: --dependency=afterok:<array>
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
echo "host=$(hostname) job=${SLURM_JOB_ID}"
"$PY" stage_a_uniqueness_aggregate.py

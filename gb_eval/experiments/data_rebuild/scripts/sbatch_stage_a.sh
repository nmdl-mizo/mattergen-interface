#!/bin/bash
#SBATCH --job-name=data_stage_a
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/stage_a_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_a_%j.err   # edit for your cluster (relative to the submit directory)
# Data rebuild, stage A (1 to 4). Submission: sbatch [--export=ALL,DATA_CNIDS=cnid_0_0] sbatch_stage_a.sh
# The target templates are narrowed with the environment variable DATA_CNIDS (comma separated). If unset, all 31 templates.
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
echo "host=$(hostname) job=${SLURM_JOB_ID} cnids=${DATA_CNIDS:-ALL}"
"$PY" stage_a_templates_dx.py

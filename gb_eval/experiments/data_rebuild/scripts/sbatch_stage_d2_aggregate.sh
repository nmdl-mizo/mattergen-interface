#!/bin/bash
#SBATCH --job-name=data_d2_agg
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/stage_d2_agg_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_d2_agg_%A_%a.err   # edit for your cluster (relative to the submit directory)
# Stages D2-3 to 5: aggregation. With DATA_D2_MODE=verify the aggregation of the re-run check (stage_d2_reverify_summary.json). With DATA_D2_SKIP_REF=1 the check of the reference rows is skipped (for small tests).
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
# The results directory is switched with the environment variable DATA_RESULTS_DIR (if unset, the production RUN/results).
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} results=${DATA_RESULTS_DIR:-default}"
ARGS=(--mode "${DATA_D2_MODE:-dedup}")
[ "${DATA_D2_SKIP_REF:-0}" = "1" ] && ARGS+=(--skip-reference-check)
"$PY" stage_d2_aggregate.py ${ARGS[@]+"${ARGS[@]}"}
if [ "${DATA_D2_MODE:-dedup}" = "dedup" ]; then
  VT=$("$PY" stage_d2_match_chunk.py plan --csv gamma_dedup.csv --match-subdir match_verify --no-reuse-pilot)
  echo "verify_tasks=${VT}"
fi

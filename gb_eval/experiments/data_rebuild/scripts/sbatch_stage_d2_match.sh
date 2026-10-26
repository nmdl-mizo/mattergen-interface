#!/bin/bash
#SBATCH --job-name=data_d2_match
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/stage_d2_match_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_d2_match_%A_%a.err   # edit for your cluster (relative to the submit directory)
# Stage D2-1: matching of candidate pairs (array). The default of r2 is no window (all pairs), 1 task = 1000 pairs (fits within MaxArraySize=30001. Production about 483 tasks).
# Submission: sbatch --array=0-<d2_tasks-1>%64 sbatch_stage_d2_match.sh (d2_tasks is the output of stage_d2_match_chunk.py plan). When using a window: DATA_WINDOW_EV=0.05; to change the number of pairs: DATA_D2_PAIRS_PER_TASK=N
# (pass the same value also to D1 build, the pilot aggregation, the D2 aggregation and the check). The production run of D2 is after the revision and re-approval of the model specification.
# Re-run check: with DATA_D2_VERIFY=1, gamma_dedup.csv is matched against match_verify/ (without reusing the check groups).
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
# The results directory is switched with the environment variable DATA_RESULTS_DIR (if unset, the production RUN/results).
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} results=${DATA_RESULTS_DIR:-default}"
ARGS=()
if [ "${DATA_D2_VERIFY:-0}" = "1" ]; then ARGS+=(--csv gamma_dedup.csv --match-subdir match_verify --no-reuse-pilot); fi
"$PY" stage_d2_match_chunk.py chunk "${SLURM_ARRAY_TASK_ID}" ${ARGS[@]+"${ARGS[@]}"}

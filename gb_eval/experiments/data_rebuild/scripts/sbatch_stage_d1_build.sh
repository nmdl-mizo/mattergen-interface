#!/bin/bash
#SBATCH --job-name=data_d1_build
#SBATCH --partition=hx1-16   # edit for your cluster
#SBATCH --nodes=1   # edit for your cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/stage_d1_build_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_d1_build_%A_%a.err   # edit for your cluster (relative to the submit directory)
# Stages D1-1 to 3: gamma_all.csv and stage_d1_summary.json. Narrowed or switched (for tests) with the environment variables DATA_CNIDS (comma separated) and DATA_INPUTS_DIR.
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
# The results directory is switched with the environment variable DATA_RESULTS_DIR (if unset, the production RUN/results).
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} results=${DATA_RESULTS_DIR:-default}"
ARGS=()
[ -n "${DATA_CNIDS:-}" ] && ARGS+=(--cnids "$DATA_CNIDS")
[ -n "${DATA_INPUTS_DIR:-}" ] && ARGS+=(--inputs-dir "$DATA_INPUTS_DIR")
"$PY" stage_d1_build_csv.py ${ARGS[@]+"${ARGS[@]}"}
echo "--- plan: pilot tasks (array 0..N-1) / D2 tasks"
PT=$("$PY" stage_d1_match_trial.py plan)    # Use an assignment so that set -e takes effect on a failure of plan
D2T=$("$PY" stage_d2_match_chunk.py plan)
echo "pilot_tasks=${PT}"
echo "d2_tasks=${D2T}"

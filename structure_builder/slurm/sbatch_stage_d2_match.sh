#!/bin/bash
# Stage D2 (array): post-relaxation duplicate matching, 1000 pairs per task.
# Submit:  sbatch --array=0-$(D2_TASKS-1)%64 sbatch_stage_d2_match.sh      (D2_TASKS from sbatch_stage_d1_build.sh)
# With the default (pilot-group reuse) run stage_d1_match_trial.py first; the examples here use --no-reuse-pilot for every group.
#SBATCH --job-name=sb_stage_d2
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$MACE_PY" "$SB_CODE/structure_builder/stageD_gamma_dedup/stage_d2_match_chunk.py" chunk "${SLURM_ARRAY_TASK_ID}" --no-reuse-pilot

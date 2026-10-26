#!/bin/bash
# Stage B: symmetric atom-pair removal (k = 1..4, 5 seeds per template and dx), bulk_membership.csv, relaxation input lists.
#SBATCH --job-name=sb_stage_b
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$MACE_PY" "$SB_CODE/structure_builder/stageB_removal/stage_b_removal.py"

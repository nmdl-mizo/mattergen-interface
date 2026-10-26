#!/bin/bash
# Stage A: 31 templates x 10 dx levels, symmetry operation and bulk refit. DATA_CNIDS=cnid_0_0,... restricts the templates (smoke test).
#SBATCH --job-name=sb_stage_a
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$MACE_PY" "$SB_CODE/structure_builder/stageA_dx/stage_a_templates_dx.py"

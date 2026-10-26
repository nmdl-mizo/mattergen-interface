#!/bin/bash
# Stage 0: build the 10x10 CNID grid of Sigma5(210)[001] bicrystals (100 POSCAR files) -> $SB_WORK/stage0/structures/g10
#SBATCH --job-name=sb_stage0
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=16           # edit for your cluster (the script uses SLURM_CPUS_PER_TASK processes)
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$MACE_PY" "$SB_CODE/structure_builder/stage0_templates/cnid_grid_sweep.py" --build-only

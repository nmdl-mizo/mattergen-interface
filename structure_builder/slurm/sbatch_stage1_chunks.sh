#!/bin/bash
# Stage 1 (array): pairwise StructureMatcher comparisons of the 100 grid structures (253 tasks of 20 comparisons).
# Submit:  sbatch --array=0-$(N-1)%24 sbatch_stage1_chunks.sh    with N = `$MACE_PY .../unified_chunk.py plan`
#SBATCH --job-name=sb_stage1
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$MACE_PY" "$SB_CODE/structure_builder/stage1_template_dedup/unified_chunk.py" chunk "${SLURM_ARRAY_TASK_ID}"

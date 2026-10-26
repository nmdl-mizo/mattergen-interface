#!/bin/bash
# Stage C (GPU array, one task per template = 210 structures): constrained relaxation with MACE-MH-1 (head omat_pbe).
# Submit:  sbatch --array=0-30%8 sbatch_stage_c_relax.sh      (DATA_RELAX_LIMIT=N relaxes only the first N structures of a template: smoke test)
# Needs the model file: MACE_MODEL_PATH (not distributed here).
#SBATCH --job-name=sb_stage_c
#SBATCH --partition=<GPU_PARTITION>            # edit for your cluster
#SBATCH --gres=shard:12              # edit for your cluster
#SBATCH --time=01:00:00              # edit for your cluster (about 38 minutes per task were measured)
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%A_%a.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"; : "${MACE_MODEL_PATH:?set MACE_MODEL_PATH}"
export PYTHONDONTWRITEBYTECODE=1
"$MACE_PY" "$SB_CODE/structure_builder/stageC_relax/stage_c_relax_chunk.py"

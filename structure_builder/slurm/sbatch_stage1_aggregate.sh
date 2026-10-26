#!/bin/bash
# Stage 1 (aggregate): union-find over the comparison results -> $SB_WORK/stage1/groups_10x10_unified.csv (31 representatives)
#SBATCH --job-name=sb_stage1_agg
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1
"$MACE_PY" "$SB_CODE/structure_builder/stage1_template_dedup/unified_aggregate.py"

#!/bin/bash
# Stage D2 (aggregate): connected components, one representative per component -> $SB_WORK/results/gamma_dedup.csv
#SBATCH --job-name=sb_stage_d2_agg
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1
"$MACE_PY" "$SB_CODE/structure_builder/stageD_gamma_dedup/stage_d2_aggregate.py" --no-reuse-pilot --skip-reference-check

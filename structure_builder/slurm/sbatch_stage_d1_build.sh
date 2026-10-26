#!/bin/bash
# Stage D1: gamma of every relaxed structure -> $SB_WORK/results/gamma_all.csv; prints the numbers of tasks of the D2 array.
#SBATCH --job-name=sb_stage_d1
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1
D="$SB_CODE/structure_builder/stageD_gamma_dedup"
"$MACE_PY" "$D/stage_d1_build_csv.py"
echo "d2_tasks=$("$MACE_PY" "$D/stage_d2_match_chunk.py" plan --no-reuse-pilot)"

#!/bin/bash
# Stage E: template-level split (25/3/3, seed 42) and the 31 template manifests; then the baseline-baseline CSV.
# Needs the repository root on the python path for stage_e_manifests.py (added by the script); SCM_HOST_Z / SCM_SOLUTE_ZS are set below.
#SBATCH --job-name=sb_stage_e
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${SB_CODE:?set SB_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
export PYTHONDONTWRITEBYTECODE=1
export SCM_HOST_Z=13 SCM_SOLUTE_ZS=
E="$SB_CODE/structure_builder/stageE_split"
"$MACE_PY" "$E/stage_e_split.py"
"$MACE_PY" "$E/stage_e_manifests.py"
"$MACE_PY" "$E/build_tilt_baseline_csv.py"

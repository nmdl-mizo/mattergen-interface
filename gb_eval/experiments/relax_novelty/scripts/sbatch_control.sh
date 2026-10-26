#!/bin/bash
# Positive control R3: 5 structures chosen at random (seed 122) from train are fixed with the 76 bulk atoms (frozen_indices of bulk_membership.csv) and
# passed through this runner to see the difference from the training label. Submit it to the GPU partition (a GPU is always used for an MLIP).
# [This file is not submitted by the author of stage P] Submission is a separate step. Usage:
#   sbatch --time=00:30:00 [--exclude=<EXCLUDE_NODES>] sbatch_control.sh        (measurement 10.8 s/structure of relaxation x 5 structures + model loading. A --time is required on the GPU partition)
# Input: verification/r3_inputs.csv (made by stage P. Columns row_index, poscar_path, frozen_indices, gamma_Jm2, area_A2, natoms)
# Output: results/control/relax_rows.csv and verification/positive_control.json (verdict PASS/FLAG. Expected: 5/5 converged, n_steps<=20, |d gamma|<=0.002 J/m2. FLAG is decided by the author)
# Environment variables: OUT_DIR (default results/control), CONTROL_JSON (default verification/positive_control.json), CONTROL_CSV (default verification/r3_inputs.csv). Refused if they exist (not overwritten).
# DRYRUN=1 bash sbatch_control.sh : nothing is run or created; the command line and the write destinations are printed and it exits.
# Execution environment: the PY, PYTHONDONTWRITEBYTECODE, cd and echo lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) are transcribed verbatim.
#
#SBATCH --job-name=al_rn_r3
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --gres=shard:12   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/r3_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/r3_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/relax_chunk.py"
OUT_DIR="${OUT_DIR:-${RUN}/results/control}"
CONTROL_JSON="${CONTROL_JSON:-${RUN}/verification/positive_control.json}"
CONTROL_CSV="${CONTROL_CSV:-${RUN}/verification/r3_inputs.csv}"
CMD=("$PY" "$SCRIPT" --control-csv "$CONTROL_CSV" --out-dir "$OUT_DIR" --control-json "$CONTROL_JSON")

echo "host=$(hostname) job=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] pre-check in the job: torch.cuda.is_available() and vram_precheck(min_free_gib=10) are done before the model loading (relax_chunk.py)"
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/relax_rows.csv  ${CONTROL_JSON}  ${RUN}/logs/r3_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ -f "$CONTROL_CSV" ] || { echo "FATAL: control csv not found: $CONTROL_CSV (stage P is required)" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

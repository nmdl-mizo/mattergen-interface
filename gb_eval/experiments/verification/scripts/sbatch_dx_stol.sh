#!/bin/bash
# Sensitivity of the instrument resolution to stol (identity decision for the 3,331 train pairs whose dx differs by 0.15 A). CPU only.
# Submission is done by the job runner. The author does not submit.
#   for s in 0.015 0.025 0.040 0.060; do sbatch --export=ALL,STOL=$s ${GBREL_CODE}/gb_eval/experiments/verification/scripts/sbatch_dx_stol.sh; done
#   Do not add --time on the CPU partition. The expected run time is the existing 53 s.
# Environment variables: STOL (required; positive number) / OUT (default VRUN/results/stol_sensitivity/stol_<STOL>/instrument_resolution_dx.json. The validation refuses if it already exists)
# DRYRUN=1 bash sbatch_dx_stol.sh : executes and creates nothing, prints the resolved command line and the write destinations, then exits.
# Execution environment: the same PY and environment variables as RUN/scripts/sbatch_n.sh:34-36 (conda mace-mh).
#
#SBATCH --job-name=al_stol_dx
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/stol_dx_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stol_dx_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
STOL="${STOL:-}"
if ! [[ "$STOL" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! awk -v s="$STOL" 'BEGIN{exit !(s>0)}'; then
  echo "FATAL: STOL must be a positive number (got '${STOL}'). Example: --export=ALL,STOL=0.040" >&2; exit 2
fi

PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
VRUN="${GBREL_WORK}/workspace/experiments/verification"
SCRIPT="${RUN}/scripts/validation_relax.py"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
OUT="${OUT:-${VRUN}/results/stol_sensitivity/stol_${STOL}/instrument_resolution_dx.json}"
CMD=("$PY" "$SCRIPT" instrument-dx --procs "$PROCS" --stol "$STOL" --out "$OUT")

echo "host=$(hostname) job=${SLURM_JOB_ID:-} stol=${STOL} out=${OUT} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT}  ${VRUN}/logs/stol_dx_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "FATAL: refusing to overwrite $OUT" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

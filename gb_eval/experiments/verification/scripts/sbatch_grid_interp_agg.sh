#!/bin/bash
# Aggregation (summary.json, summary.md, fig_dmin_hist.png, fig_gbdx_delta.png). Run it with afterok after all 23 tasks (submitted by the job runner).
#   sbatch --dependency=afterok:<array job ID> sbatch_grid_interp_agg.sh   (no --time)
# Environment variables: OUT_DIR (default results/grid_interp) / DRYRUN=1 (executes and creates nothing; prints)
#SBATCH --job-name=al_gi_agg
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/grid_interp_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/grid_interp_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac

PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

VRUN="${GBREL_WORK}/workspace/experiments/verification"
SCRIPT="${VRUN}/scripts/grid_interp.py"
OUT_DIR="${OUT_DIR:-${VRUN}/results/grid_interp}"
CMD=("$PY" "$SCRIPT" aggregate --out-dir "$OUT_DIR")

echo "host=$(hostname) job=${SLURM_JOB_ID:-} out_dir=${OUT_DIR}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{summary.json,summary.md,fig_dmin_hist.png,fig_gbdx_delta.png,AGG.DONE}  ${VRUN}/logs/grid_interp_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi
[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
cd "${VRUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

#!/bin/bash
# Nearest-neighbour matching against train (L->T) and mutual matching of the 10 low-gamma novel proposed structures of Al. 20 CPUs, 1 job (submitted by the job runner).
#   sbatch sbatch_low_gamma_nn.sh   (no --time)
# Environment variables: OUT_DIR (default results/low_gamma_novel_view/nn) / DRYRUN=1 (executes and creates nothing; prints only the write destinations)
#SBATCH --job-name=al_lg_nn
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/low_gamma_nn_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/low_gamma_nn_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac

PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

VRUN="${GBREL_WORK}/workspace/experiments/verification"
SCRIPT="${VRUN}/scripts/low_gamma_nn.py"
OUT_DIR="${OUT_DIR:-${VRUN}/results/low_gamma_novel_view/nn}"
NPROCS="${SLURM_CPUS_PER_TASK:-20}"
CMD=("$PY" "$SCRIPT" --out-dir "$OUT_DIR" --procs "$NPROCS")

echo "host=$(hostname) job=${SLURM_JOB_ID:-} out_dir=${OUT_DIR} procs=${NPROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{nn_L.csv,pairs_L.csv,summary.md,summary.json,NN_L.DONE}  ${VRUN}/logs/low_gamma_nn_<jobname>_<jobid>.{out,err}"
  for f in nn_L.csv pairs_L.csv summary.md summary.json NN_L.DONE; do
    if [ -e "${OUT_DIR}/${f}" ]; then echo "[DRYRUN] WARNING: already exists (script will stop): ${OUT_DIR}/${f}"; fi
  done
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi
[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
for f in nn_L.csv pairs_L.csv summary.md summary.json NN_L.DONE; do
  if [ -e "${OUT_DIR}/${f}" ]; then echo "FATAL: output already exists, refusing to overwrite: ${OUT_DIR}/${f}" >&2; exit 2; fi
done
cd "${VRUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

#!/bin/bash
# Positive and negative controls of the novelty instrument (pymatgen StructureMatcher). CPU only. No MLIP or GPU is used.
# Plan: see the experiment description.
# [The author of this file does not submit] Submission is by the job runner (with dependencies). The environment variable STAGE switches S/R/A:
#   S: sbatch --job-name=al_nc_S --export=ALL,STAGE=S sbatch_novelty_control.sh                                   (only writes bases.csv)
#   R: sbatch --job-name=al_nc_R --array=0-19%6 --dependency=afterok:<S> --export=ALL,STAGE=R sbatch_novelty_control.sh  (1 task = 1 base structure)
#   A: sbatch --job-name=al_nc_A --dependency=afterok:<R> --export=ALL,STAGE=A sbatch_novelty_control.sh           (aggregation + gate)
#   Do not add --time on the CPU partition. The model-specification check is not called (only the instrument check on evaluated products).
# Environment variables: STAGE (required; S|R|A) / DRYRUN (0|1) / TASK_ID (only for DRYRUN; the base structure ID of STAGE=R) / OUT_DIR (default results/novelty_control)
# DRYRUN=1 STAGE=R TASK_ID=0 bash sbatch_novelty_control.sh : executes and creates nothing; prints the command line, the existence of the inputs and the write destinations, then exits.
# Execution environment: the PY, PYTHONDONTWRITEBYTECODE and OMP lines of sbatch_n.sh (${GBREL_CODE}/gb_eval/experiments/relax_novelty/scripts/) copied verbatim
#   (conda mace-mh. pymatgen 2025.6.14 is the same environment as the production novelty matching).
# Completion: each stage prints "done rc=0 stage=<STAGE>" at its end (acceptance: the log contains 'done rc=0').
#   The product of R is DONE_base<kk> (written by python after checking that there are 15 rows). The job runner confirms 20 of them and 300 rows in total.
# Exit code: 0 = success / 2 = failure of the inputs or checks (including fewer than 15 rows, a degeneracy check of the construction rms>1e-6, and a canonicalize exception)
#
#SBATCH --job-name=al_novctl
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/novctl_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/novctl_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
STAGE="${STAGE:-}"
case "$STAGE" in S|R|A) ;; *) echo "FATAL: STAGE must be S|R|A (got '${STAGE}')" >&2; exit 2 ;; esac

# ---- verbatim from the source (sbatch_n.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # the matching uses one thread per process (parallelized with multiprocessing)

VRUN="${GBREL_WORK}/workspace/experiments/verification"
RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SC="${VRUN}/scripts"
OUT_DIR="${OUT_DIR:-${VRUN}/results/novelty_control}"
PROCS="${SLURM_CPUS_PER_TASK:-5}"

TID=""
if [ "$STAGE" = "R" ]; then
  TID="${SLURM_ARRAY_TASK_ID:-}"
  if [ -z "$TID" ] && [ "$DRYRUN" = "1" ]; then TID="${TASK_ID:-}"; fi
  [[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: STAGE=R needs SLURM_ARRAY_TASK_ID (sbatch --array=0-19%6). For DRYRUN use TASK_ID=<k>" >&2; exit 2; }
  [ "$TID" -le 19 ] || { echo "FATAL: base id must be 0..19 (got ${TID})" >&2; exit 2; }
fi

case "$STAGE" in
  S) CMD=("$PY" "${SC}/novelty_control.py" --select --out-dir "$OUT_DIR")
     W_TARGETS="${OUT_DIR}/bases.csv" ;;
  R) CMD=("$PY" "${SC}/novelty_control.py" --base-id "$TID" --out-dir "$OUT_DIR" --procs "$PROCS")
     W_TARGETS="${OUT_DIR}/rows_base$(printf '%02d' "$TID").csv ${OUT_DIR}/DONE_base$(printf '%02d' "$TID")" ;;
  A) CMD=("$PY" "${SC}/aggregate_novelty_control.py" --out-dir "$OUT_DIR")
     W_TARGETS="${OUT_DIR}/cells.csv ${OUT_DIR}/control_check.json ${OUT_DIR}/summary.md" ;;
esac

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-} stage=${STAGE} task=${TID} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${W_TARGETS}  ${VRUN}/logs/novctl_<jobname>_<jobid>.{out,err}"
  for f in "$PY" "${SC}/novelty_control.py" "${SC}/aggregate_novelty_control.py" "${RUN}/scripts/common.py" \
           "${RUN}/results/train_ref_manifest.csv"; do
    if [ -e "$(rp "$f")" ]; then echo "[DRYRUN] exists: $f"; else echo "[DRYRUN][WARN] not found on this node: $f"; fi
  done
  if [ "$STAGE" = "R" ]; then
    if [ -f "$(rp "${OUT_DIR}/bases.csv")" ]; then echo "[DRYRUN] bases.csv exists: ${OUT_DIR}/bases.csv"; else echo "[DRYRUN][WARN] bases.csv not found (made by stage S. FATAL in production): ${OUT_DIR}/bases.csv"; fi
  fi
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "${SC}/novelty_control.py" ] || { echo "FATAL: script not found: ${SC}/novelty_control.py" >&2; exit 2; }
[ -f "${SC}/aggregate_novelty_control.py" ] || { echo "FATAL: script not found: ${SC}/aggregate_novelty_control.py" >&2; exit 2; }
[ -f "${RUN}/scripts/common.py" ] || { echo "FATAL: RUN common.py not found: ${RUN}/scripts/common.py" >&2; exit 2; }
[ -f "${RUN}/results/train_ref_manifest.csv" ] || { echo "FATAL: train_ref_manifest.csv not found" >&2; exit 2; }
mkdir -p "$OUT_DIR"
cd "${SC}"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"
echo "done rc=0 stage=${STAGE}"

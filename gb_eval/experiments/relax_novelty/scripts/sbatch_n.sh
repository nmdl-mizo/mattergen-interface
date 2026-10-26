#!/bin/bash
# Novelty matching (N-pre, N-post; 1 array task = 1 chunk). CPU only (CPU partition, multiprocessing inside the task). No MLIP is used.
# [This file is not submitted by the author of stage P] Submission is a separate step (it may be submitted together with R1 with a dependency):
#   N-additional: sbatch --array=0-111%6 --dependency=afterok:<array ID of R1-additional> --export=ALL,LAYER=additional sbatch_n.sh   (all 112 chunks; the 2 chunks of R0 are already done)
#   N-main: sbatch --array=0-27%6  --dependency=afterok:<array ID of R-main> --export=ALL,LAYER=main sbatch_n.sh
#   Do not give --time on the CPU partition. %6 is a guide for the number of simultaneous tasks so that the CPU partition (at most 10 nodes, 24 CPUs) is not monopolized (adjust as needed).
# Environment variables: LAYER (required: additional|main) / OUT_ROOT (default results/novelty) / RELAX_ROOT (default results/relax) / PHASE (default both. pre|post|both) / TASK_ID (for DRYRUN only)
# DRYRUN=1 bash sbatch_n.sh : does not run or create anything; prints the resolution of the task (the row of array_map), the command line and the output destinations, and exits.
# Execution environment: the PY and PYTHONDONTWRITEBYTECODE lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) transcribed verbatim
#   (conda evaluation environment. pymatgen 2025.6.14 is the same environment as the matching of data-rebuild stage D2).
# Idempotent: completed tasks (row file + marker) are skipped with exit 0. N-post exits with 2 if RELAX_DONE is missing.
# Measured cost (validation n1u1): one matching (no match) takes about 0.45 s (N1(b): 20 structures x at most 1191 pairs with 20 processes in 540 s). One chunk, one phase = 25 structures x about 1,100 pairs x 0.45 s = about 3.4 CPU hours.
#   Parallelism is per structure, so 25 structures = 5 CPUs x 5 waves splits without waste (--cpus-per-task=5; 4 tasks on a 24-core node). The wall time of one task is about 2.5 to 3 hours for pre+post (if every chunk is treated as novel and scanned in full).
#
#SBATCH --job-name=al_rn_nov
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/novelty_%x_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/novelty_%x_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-}"
case "$LAYER" in additional|main) ;; *) echo "FATAL: LAYER must be additional or main (got '${LAYER}')" >&2; exit 2 ;; esac
PHASE="${PHASE:-both}"
case "$PHASE" in pre|post|both) ;; *) echo "FATAL: PHASE must be pre|post|both (got '${PHASE}')" >&2; exit 2 ;; esac
TID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TID" ] && [ "$DRYRUN" = "1" ]; then TID="${TASK_ID:-}"; fi
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: SLURM_ARRAY_TASK_ID is not set (sbatch --array=... is required. For DRYRUN use TASK_ID=<k>)" >&2; exit 2; }

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # the matching uses one thread per process (parallelized with multiprocessing)

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/novelty_chunk.py"
OUT_ROOT="${OUT_ROOT:-${RUN}/results/novelty}"
RELAX_ROOT="${RELAX_ROOT:-${RUN}/results/relax}"
ARRAY_MAP="${RUN}/results/array_map_${LAYER}.csv"
PROCS="${SLURM_CPUS_PER_TASK:-5}"
CMD=("$PY" "$SCRIPT" --layer "$LAYER" --task-id "$TID" --phase "$PHASE" --procs "$PROCS" --out-dir "${OUT_ROOT}/${LAYER}" --relax-dir "${RELAX_ROOT}/${LAYER}")

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-} task=${TID} layer=${LAYER} phase=${PHASE} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  if [ -f "$(rp "$ARRAY_MAP")" ]; then echo "[DRYRUN] array_map row (header + task ${TID}): $(grep -E "^(array_index|${TID})," "$(rp "$ARRAY_MAP")" | tr "\n" " ")"; else echo "[DRYRUN][WARN] array map not found (FATAL in production. Made by stage P --layers ${LAYER}): ${ARRAY_MAP}"; fi
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT_ROOT}/${LAYER}/<model>_w<W>/chunk<cc>/{novelty_pre_rows.csv,NOVELTY_PRE_DONE,novelty_post_rows.csv,NOVELTY_POST_DONE}  ${RUN}/logs/novelty_<jobname>_<jobid>_<k>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ -f "$ARRAY_MAP" ] || { echo "FATAL: array map not found: $ARRAY_MAP (stage P is required)" >&2; exit 2; }
[ -f "${RUN}/results/train_ref_manifest.csv" ] || { echo "FATAL: train_ref_manifest.csv not found" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

#!/bin/bash
# Novelty matching (N-pre and N-post; 1 array task = 1 chunk). VRUN copy. CPU only (multiprocessing within a task). No MLIP is used.
# [This file is not submitted by the author] Submission is by the job runner via submit_layer_chain.sh:
#   sbatch --job-name=al_novelty_<layer> --array=0-13%6 --dependency=afterok:<relax> --export=ALL,LAYER=<layer> sbatch_n.sh
#   Do not add --time on the CPU partition. %6 is a guide for the number of concurrent tasks (the parent may adjust it).
# Environment variables: LAYER (required; a regeneration layer of layers.json. main is read only, so not allowed) / OUT_ROOT (default results/novelty) / RELAX_ROOT (default results/relax) / PHASE (default both; pre|post|both) / TASK_ID (only for DRYRUN)
# DRYRUN=1 bash sbatch_n.sh : does not run or create anything; prints the resolution of the task (the row of array_map), the command line and the output destinations, and exits.
# Execution environment: the PY and PYTHONDONTWRITEBYTECODE lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) transcribed verbatim
#   (conda evaluation environment. pymatgen 2025.6.14 is the same environment as the matching of data-rebuild stage D2).
# Idempotent: completed tasks (row file + marker) are skipped with exit 0. N-post exits with 2 if RELAX_DONE is missing.
# Measured cost (validation n1u1): one matching (no match) takes about 0.45 s (N1(b): 20 structures x at most 1191 pairs with 20 processes in 540 s). One chunk, one phase = 25 structures x about 1,100 pairs x 0.45 s = about 3.4 CPU hours.
#   Parallelism is per structure, so 25 structures = 5 CPUs x 5 waves splits without waste (--cpus-per-task=5; 4 tasks on a 24-core node). The wall time of one task is about 2.5 to 3 hours for pre+post (if every chunk is treated as novel and scanned in full).
#
#SBATCH --job-name=al_novelty   # overridden at submission with --job-name=al_novelty_<layer>
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/novelty_%x_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/novelty_%x_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-}"
SC="${GBREL_CODE}/gb_eval/experiments/verification/scripts"
[ -d "$SC" ] || SC="/mn""t${SC}"
source "${SC}/layer_lib.sh"
layer_require "$LAYER"
PHASE="${PHASE:-both}"
case "$PHASE" in pre|post|both) ;; *) echo "FATAL: PHASE must be pre|post|both (got '${PHASE}')" >&2; exit 2 ;; esac
TID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TID" ] && [ "$DRYRUN" = "1" ]; then TID="${TASK_ID:-}"; fi
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: SLURM_ARRAY_TASK_ID is not set (sbatch --array=... is required. For DRYRUN use TASK_ID=<k>)" >&2; exit 2; }

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # the matching uses one thread per process (parallelized with multiprocessing)

RUN="${GBREL_WORK}/workspace/experiments/verification"   # VRUN version (output, manifest, array_map and scripts are in VRUN)
RUN_TRAIN="${GBREL_WORK}/workspace/experiments/relax_novelty"   # The existing train_ref_manifest of RUN is used read only as the comparison partner of novelty (not rebuilt for each layer)
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
[ -f "${RUN_TRAIN}/results/train_ref_manifest.csv" ] || { echo "FATAL: train_ref_manifest.csv not found" >&2; exit 2; }
"$PY" "${RUN}/scripts/stage_gate.py" --layer "$LAYER" --stage relax   # Entrance check (products of the preceding stage: 350)
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

#!/bin/bash
# N(s): sensitivity of the novelty matching to stol (stol of StructureMatcher is overridden by $STOL). A copy of sbatch_n.sh of the RUN version + STOL.
# Submission is done by the job runner. The author does not submit.
#   sbatch --array=0-27%8 --export=ALL,STOL=<s>,LAYER=main ${GBREL_CODE}/gb_eval/experiments/verification/scripts/sbatch_n_stol.sh
#   Do not add --time on the CPU partition. %N is decided right before submission by measuring the free CPUs of the nodes.
# Environment variables: STOL (required; positive number. Example 0.040) / LAYER (required; main only) / OUT_ROOT (default VRUN/results/stol_sensitivity/stol_<STOL>/novelty.
#   --out-dir is ${OUT_ROOT}/${LAYER}, so the per-structure CSVs appear in ${OUT_ROOT}/main/<model>_w2.0/chunkNN/) / RELAX_ROOT (default RUN/results/relax) / PHASE (default both) / TASK_ID (only for DRYRUN)
# --novelty-root of the aggregation sbatch_a_stol.sh is ${OUT_ROOT} (= .../novelty). aggregate reads <root>/<layer>/<armw>/<chunk> (aggregate_relax_novelty.py:127,136).
# DRYRUN=1 bash sbatch_n_stol.sh : executes and creates nothing, prints the resolved command line and the write destinations, then exits.
# Execution environment: the PY and environment variables of RUN/scripts/sbatch_n.sh:34-36 copied verbatim (conda mace-mh; proven in earlier jobs).
#
#SBATCH --job-name=al_stol_nov
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=5
#SBATCH --output=logs/stol_nov_%x_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stol_nov_%x_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
STOL="${STOL:-}"
if ! [[ "$STOL" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! awk -v s="$STOL" 'BEGIN{exit !(s>0)}'; then
  echo "FATAL: STOL must be a positive number (got '${STOL}'). Example: --export=ALL,STOL=0.040" >&2; exit 2
fi
LAYER="${LAYER:-}"
case "$LAYER" in main) ;; *) echo "FATAL: LAYER must be main (got '${LAYER}'). The sensitivity of additional is out of scope" >&2; exit 2 ;; esac
PHASE="${PHASE:-both}"
case "$PHASE" in pre|post|both) ;; *) echo "FATAL: PHASE must be pre|post|both (got '${PHASE}')" >&2; exit 2 ;; esac
TID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TID" ] && [ "$DRYRUN" = "1" ]; then TID="${TASK_ID:-}"; fi
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: SLURM_ARRAY_TASK_ID is not set (sbatch --array=... is required. For DRYRUN use TASK_ID=<k>)" >&2; exit 2; }

PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
VRUN="${GBREL_WORK}/workspace/experiments/verification"
SCRIPT="${RUN}/scripts/novelty_chunk.py"
OUT_ROOT="${OUT_ROOT:-${VRUN}/results/stol_sensitivity/stol_${STOL}/novelty}"
RELAX_ROOT="${RELAX_ROOT:-${RUN}/results/relax}"
ARRAY_MAP="${RUN}/results/array_map_${LAYER}.csv"
PROCS="${SLURM_CPUS_PER_TASK:-5}"
CMD=("$PY" "$SCRIPT" --layer "$LAYER" --task-id "$TID" --phase "$PHASE" --procs "$PROCS" --out-dir "${OUT_ROOT}/${LAYER}" --relax-dir "${RELAX_ROOT}/${LAYER}" --stol "$STOL")

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-} task=${TID} layer=${LAYER} phase=${PHASE} procs=${PROCS} stol=${STOL}"
if [ "$DRYRUN" = "1" ]; then
  if [ -f "$(rp "$ARRAY_MAP")" ]; then echo "[DRYRUN] array_map row (header + task ${TID}): $(grep -E "^(array_index|${TID})," "$(rp "$ARRAY_MAP")" | tr "\n" " ")"; else echo "[DRYRUN][WARN] array map not found: ${ARRAY_MAP}"; fi
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT_ROOT}/${LAYER}/<model>_w<W>/chunk<cc>/{novelty_pre_rows.csv,NOVELTY_PRE_DONE,novelty_post_rows.csv,NOVELTY_POST_DONE}  ${VRUN}/logs/stol_nov_<jobname>_<jobid>_<k>.{out,err}"
  echo "[DRYRUN] --novelty-root for aggregate: ${OUT_ROOT}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ -f "$ARRAY_MAP" ] || { echo "FATAL: array map not found: $ARRAY_MAP" >&2; exit 2; }
[ -f "${RUN}/results/train_ref_manifest.csv" ] || { echo "FATAL: train_ref_manifest.csv not found" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

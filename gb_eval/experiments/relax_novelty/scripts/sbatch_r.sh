#!/bin/bash
# Stage R: relaxation (1 array task = 1 chunk = 25 structures). Submit to the GPU partition (a GPU is always used for an MLIP).
# [This file is not submitted by the author of stage P] Submission is a separate step (after the acceptance of R0). --array and --time are intentionally not written in the header:
#   R0 (proposed w3.0 chunk00 and baseline w3.0 chunk00 of additional):
#     sbatch --array=28,84 --time=01:00:00 --export=ALL,LAYER=additional [--exclude=<EXCLUDE_NODES>] sbatch_r.sh      (28 and 84 are confirmed in results/array_map_additional.csv)
#   R1-additional (the remaining 110): sbatch --array=$(cat ${GBREL_WORK}/workspace/experiments/relax_novelty/results/r1_additional_array_spec.txt) --time=<maximum duration of one task of R0 x 1.5> --export=ALL,LAYER=additional sbatch_r.sh
#   R-main (28 jobs): sbatch --array=0-27 --time=... --export=ALL,LAYER=main sbatch_r.sh   (after the manifest of main is made)
#   A --time is required on the GPU partition. Run the GPU pre-check and the job pre-check before submission.
# Environment variables: LAYER (required additional|main) / OUT_ROOT (default results/relax. The output is <OUT_ROOT>/<LAYER>/<model>_w<W>/chunk<cc>/) / FIX_INDICES (default none = default per arm: proposed=0..75, baseline=none.
#   The FAIL validation of check R0 is run on proposed with FIX_INDICES=none and OUT_ROOT=<another dir> -> exit 4 because the displacement of the fixed atoms is > 0) / TASK_ID (for DRYRUN only. Instead of SLURM_ARRAY_TASK_ID)
# DRYRUN=1 bash sbatch_r.sh : nothing is run or created; the resolution of the task (the row of array_map), the command line and the write destinations are printed and it exits (python is not used on the login node).
# Execution environment: the PY, PYTHONDONTWRITEBYTECODE, cd and echo lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) are transcribed verbatim.
# Idempotent: completed chunks are skipped by relax_chunk.py with exit 0 (the afterok chain is not cut). If the output dir exists and RELAX_DONE does not, exit 2.
# shard: a relaxation task has 120 atoms; following the record of the evaluation of gamma_pre (shard:16) and the relaxation (shard:12, sbatch_stage_c_relax.sh), shard:12. The VRAM peak is measured with R0 and R1 is fixed (table of the plan).
#
#SBATCH --job-name=al_rn_relax
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --gres=shard:12   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/relax_%x_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/relax_%x_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-}"
case "$LAYER" in additional|main) ;; *) echo "FATAL: LAYER must be additional or main (got '${LAYER}')" >&2; exit 2 ;; esac
TID="${SLURM_ARRAY_TASK_ID:-}"
if [ -z "$TID" ] && [ "$DRYRUN" = "1" ]; then TID="${TASK_ID:-}"; fi
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: SLURM_ARRAY_TASK_ID is not set (sbatch --array=... is required. For DRYRUN use TASK_ID=<k>)" >&2; exit 2; }

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/relax_chunk.py"
OUT_ROOT="${OUT_ROOT:-${RUN}/results/relax}"
ARRAY_MAP="${RUN}/results/array_map_${LAYER}.csv"
CMD=("$PY" "$SCRIPT" --layer "$LAYER" --task-id "$TID" --out-dir "${OUT_ROOT}/${LAYER}")
if [ -n "${FIX_INDICES:-}" ]; then CMD+=(--fix-indices "$FIX_INDICES"); fi

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-} task=${TID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} layer=${LAYER} out_root=${OUT_ROOT} fix_indices=${FIX_INDICES:-<arm default>}"
if [ "$DRYRUN" = "1" ]; then
  if [ -f "$(rp "$ARRAY_MAP")" ]; then echo "[DRYRUN] array_map row (header + task ${TID}): $(grep -E "^(array_index|${TID})," "$(rp "$ARRAY_MAP")" | tr "\n" " ")"; else echo "[DRYRUN][WARN] array map not found (FATAL in production. Made by stage P --layers ${LAYER}): ${ARRAY_MAP}"; fi
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] pre-check in the job: relax_chunk.py checks torch.cuda.is_available() (an MLIP is not run on CPU) and runner_guards vram_precheck(min_free_gib=10) before the first structure"
  echo "[DRYRUN] WRITE TARGETS: ${OUT_ROOT}/${LAYER}/<model>_w<W>/chunk<cc>/{relaxed.extxyz,relax_rows.csv,RELAX_DONE}  ${RUN}/logs/relax_<jobname>_<jobid>_<k>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ -f "$ARRAY_MAP" ] || { echo "FATAL: array map not found: $ARRAY_MAP (stage P is required)" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

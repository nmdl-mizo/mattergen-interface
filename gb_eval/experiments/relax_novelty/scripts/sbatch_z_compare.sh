#!/bin/bash
# Comparison of the noise-floor stage (noise_floor_compare.py): matches |d gamma| of the original evaluation and the two recomputations by (chunk, cif_index) and writes noise_floor_summary*.json. CPU only.
#   Submission: sbatch --export=ALL,LAYERS=additional sbatch_z_compare.sh        (additional|main|both. both = noise_floor_summary.json after all 8 are present. No --time on the CPU partition)
#   Check: sbatch --export=ALL,SELF_CHECK=1 sbatch_z_compare.sh         (checks the matching, the statistics and the FAIL detection of Z1 with synthetic replicates -> verification/noise_compare_self_check/)
# Check Z1: each csv has 350 rows and all rows are matched with the original by key (0 unmatched). If violated, exit 2 without writing the output.
# Execution environment: the PY and PYTHONDONTWRITEBYTECODE lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) are transcribed verbatim.
# DRYRUN=1 bash sbatch_z_compare.sh : nothing is run or created; the command line and the write destinations are printed and it exits.
#
#SBATCH --job-name=al_rn_zcmp
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/zcmp_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/zcmp_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
SELF_CHECK="${SELF_CHECK:-0}"
case "$SELF_CHECK" in 0|1) ;; *) echo "FATAL: SELF_CHECK must be 0 or 1 (got '${SELF_CHECK}')" >&2; exit 2 ;; esac
LAYERS="${LAYERS:-}"
if [ "$SELF_CHECK" = "0" ]; then case "$LAYERS" in additional|main|both) ;; *) echo "FATAL: LAYERS must be additional|main|both (got '${LAYERS}')" >&2; exit 2 ;; esac; fi

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/noise_floor_compare.py"
if [ "$SELF_CHECK" = "1" ]; then CMD=("$PY" "$SCRIPT" --self_check --self_check-dir "${SELF_CHECK_DIR:-${RUN}/verification/noise_compare_self_check}"); TARGET="${SELF_CHECK_DIR:-${RUN}/verification/noise_compare_self_check}/";
else CMD=("$PY" "$SCRIPT" --layers "$LAYERS"); if [ "$LAYERS" = both ]; then TARGET="${RUN}/results/noise/noise_floor_summary.json"; else TARGET="${RUN}/results/noise/noise_floor_summary_${LAYERS}.json"; fi; fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-} layers=${LAYERS:-<self_check>} self_check=${SELF_CHECK}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${TARGET}  ${RUN}/logs/zcmp_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi
[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

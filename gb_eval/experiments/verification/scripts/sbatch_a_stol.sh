#!/bin/bash
# A(s): sensitivity of the aggregation of connected components (number of groups) and metrics to stol. A copy of sbatch_a.sh of the RUN version + STOL. CPU only (ix14-23).
# Submission is done by the job runner. The author does not submit.
#   sbatch --dependency=afterok:<job ID of the array of N(s)> --export=ALL,STOL=<s> ${GBREL_CODE}/gb_eval/experiments/verification/scripts/sbatch_a_stol.sh
#   Do not add --time on ix14-23.
# Environment variables: STOL (required; positive number) / NOVELTY_ROOT (default VRUN/results/stol_sensitivity/stol_<STOL>/novelty. The same as OUT_ROOT of sbatch_n_stol.sh.
#   RUN/results/novelty may be given for the acceptance check) / OUT_DIR (default VRUN/results/stol_sensitivity/stol_<STOL>/agg)
# The aggregation is fixed to --layers main --selected proposed:2.0,baseline:2.0 (passing SELECTED as a comma-separated --export gets cut, so it is fixed here. 724488).
# DRYRUN=1 bash sbatch_a_stol.sh : executes and creates nothing, prints the resolved command line and the write destinations, then exits.
# Execution environment: PY and environment variables copied verbatim from RUN/scripts/sbatch_a.sh:46-48 (conda mace-mh).
#
#SBATCH --job-name=al_stol_agg
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/stol_agg_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stol_agg_%x_%j.err   # edit for your cluster (relative to the submit directory)

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
SCRIPT="${RUN}/scripts/aggregate_relax_novelty.py"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
NOVELTY_ROOT="${NOVELTY_ROOT:-${VRUN}/results/stol_sensitivity/stol_${STOL}/novelty}"
OUT_DIR="${OUT_DIR:-${VRUN}/results/stol_sensitivity/stol_${STOL}/agg}"
CMD=("$PY" "$SCRIPT" --layers main --selected "proposed:2.0,baseline:2.0" --novelty-root "$NOVELTY_ROOT" --out-dir "$OUT_DIR" --procs "$PROCS" --stol "$STOL")

echo "host=$(hostname) job=${SLURM_JOB_ID:-} stol=${STOL} novelty_root=${NOVELTY_ROOT} out_dir=${OUT_DIR} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] READ  (hierarchy read by aggregate): ${NOVELTY_ROOT}/main/<model>_w2.0/chunk<cc>/{novelty_pre_rows.csv,NOVELTY_PRE_DONE,novelty_post_rows.csv,NOVELTY_POST_DONE}  (aggregate_relax_novelty.py:127,136)"
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{relax_novelty_summary.json,relax_novelty_per_target.csv,relax_novelty_report.md,figs/relax_novelty_grid.png,figs/relax_novelty_grid.pdf}  ${VRUN}/logs/stol_agg_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

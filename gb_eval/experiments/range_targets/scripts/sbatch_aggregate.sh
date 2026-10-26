#!/bin/bash
# Copy of the range-target version. Source: ${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/sbatch_aggregate.sh.
# Only RUN_DIR (the value of the RUN variable), TARGETS_JSON, CARD_HASH12 and the range of k (14..27) were changed.
# A thin wrapper that runs stage A (aggregation and figures of the tracking) on a compute node (CPU).
#   Usage: sbatch [--dependency=afterok:<all job IDs of E joined with ':'>] sbatch_aggregate.sh
#   Overriding the output destination and the log (example of writing the 8 combinations of w1 to 4 separately from the existing products of w=1 to 3):
#     sbatch --export=ALL,OUT_DIR=<RUN_DIR>/results/w1to4,AGG_LOG=<RUN_DIR>/verification/A_aggregate_w1to4.log sbatch_aggregate.sh
#     If unspecified, as before OUT_DIR=<RUN_DIR>/results, AGG_LOG=<RUN_DIR>/verification/A_aggregate.log.
#     OUT_DIR may only be an absolute path under <RUN_DIR>/results and AGG_LOG only an absolute path under <RUN_DIR>/verification/ (".." is not allowed; a violation exits with 2).
#   DRYRUN=1 bash sbatch_aggregate.sh ... prints the command line and the output destinations and exits (creates and runs nothing).
# Execution environment: the evaluation environment (numpy, scipy, pandas, matplotlib. The import was confirmed on the compute nodes of the CPU partition). No GPU is used.
# Input: RUN/results/eval/<model>_w<W>/{gamma_pre.csv,validation.json} (both arms for each existing w. 6 files for w=1 to 3, 8 files with the extension w=4.0. w is determined automatically by aggregate_tracking.py from the existence in eval-root).
# Output (OUT_DIR default): RUN/results/{tracking_summary.json,tracking_per_target.csv,tracking_report.md,figs/}; standard output is also teed to RUN/verification/A_aggregate.log (refused if it exists).
#
#SBATCH --job-name=al_aggregate
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/aggregate_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/aggregate_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1

RUN_DIR="${GBREL_WORK}/workspace/experiments/range_targets"
AGG_PY="${RUN_DIR}/scripts/aggregate_tracking.py"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/results}"
AGG_LOG="${AGG_LOG:-${RUN_DIR}/verification/A_aggregate.log}"

case "$OUT_DIR" in
    *..*) echo "FATAL: OUT_DIR must not contain '..' (got '${OUT_DIR}')" >&2; exit 2 ;;
    "${RUN_DIR}/results"|"${RUN_DIR}/results/"*) ;;
    *) echo "FATAL: OUT_DIR must be ${RUN_DIR}/results or an absolute path under ${RUN_DIR}/results/ (got '${OUT_DIR}')" >&2; exit 2 ;;
esac
case "$AGG_LOG" in
    *..*) echo "FATAL: AGG_LOG must not contain '..' (got '${AGG_LOG}')" >&2; exit 2 ;;
    "${RUN_DIR}/verification/"*) ;;
    *) echo "FATAL: AGG_LOG must be an absolute path under ${RUN_DIR}/verification/ (got '${AGG_LOG}')" >&2; exit 2 ;;
esac

echo "host: $(hostname)"
echo "DRYRUN=${DRYRUN}"
if [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN] source ${CONDA_ROOT}/etc/profile.d/conda.sh; conda activate mace-mh"
    echo "[DRYRUN] python ${AGG_PY} --eval-root ${RUN_DIR}/results/eval --out-dir ${OUT_DIR}   | tee ${AGG_LOG}"
    echo "[DRYRUN] WRITE TARGETS (all under ${RUN_DIR}/): ${OUT_DIR}/{tracking_summary.json,tracking_per_target.csv,tracking_report.md,figs/} ${AGG_LOG} ${RUN_DIR}/logs/aggregate_<jobid>.{out,err}"
    echo "[DRYRUN] done (nothing executed, nothing created)"
    exit 0
fi

[ -f "$AGG_PY" ] || { echo "FATAL: not found: $AGG_PY" >&2; exit 2; }
[ ! -e "$AGG_LOG" ] || { echo "FATAL: log exists (no overwrite): $AGG_LOG" >&2; exit 2; }
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate mace-mh
echo "start: $(date -u)"
python "$AGG_PY" --eval-root "${RUN_DIR}/results/eval" --out-dir "${OUT_DIR}" 2>&1 | tee "$AGG_LOG"
echo "end: $(date -u)"

#!/bin/bash
# Stage U+A: uniqueness decision (connected components), aggregation of the metrics, additional items, figures and report. CPU only. No MLIP is used.
# [This file is not submitted by the author of stage P] Submission is a separate step:
#   U+A (interim aggregation of additional): sbatch --dependency=afterok:<array ID of N-additional> --export=ALL,LAYERS=additional sbatch_a.sh      -> results/interim_additional/ (checks 2,800 rows. No main comparison section)
#   U+A (final aggregation):        SELECTED=proposed:2.0,baseline:2.0 sbatch --dependency=afterok:<N-additional>:<N-main> --export=ALL,LAYERS=both sbatch_a.sh   -> results/ (checks 3,500 rows)
#   [Note] If SELECTED is written as --export=ALL,...,SELECTED=a,b it is cut at the comma and baseline is dropped. Always pass it as an environment variable.
#   Give no --time on the CPU partition. SELECTED only with LAYERS=both (decision note).
# Environment variables: LAYERS (required additional|main|both) / SELECTED (optional. Example proposed:2.0,baseline:2.0) / OUT_DIR (default per LAYERS: additional=results/interim_additional, main=results/interim_main, both=results)
# DRYRUN=1 bash sbatch_a.sh : nothing is run or created; the command line and the write destinations are printed and it exits.
# Execution environment: the PY and PYTHONDONTWRITEBYTECODE lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) transcribed verbatim
#   (the evaluation environment. numpy, matplotlib, pymatgen).
#
#SBATCH --job-name=al_rn_agg
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/agg_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/agg_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYERS="${LAYERS:-}"
case "$LAYERS" in additional|main|both) ;; *) echo "FATAL: LAYERS must be additional|main|both (got '${LAYERS}')" >&2; exit 2 ;; esac
SELECTED="${SELECTED:-}"
if [ -n "$SELECTED" ] && [ "$LAYERS" = "additional" ]; then echo "FATAL: SELECTED is not allowed with LAYERS=additional (the interim aggregation does not make the main comparison section)" >&2; exit 2; fi
if [ -n "$SELECTED" ]; then
  _sel_re='^(proposed|baseline):[0-9]+(\.[0-9]+)?$'
  _sel_ok=1; _sel_n=0; _sel_proposed=0; _sel_van=0
  if [[ "$SELECTED" == *, ]]; then _sel_ok=0; fi
  IFS=',' read -r -a _sel_items <<< "$SELECTED" || true
  for _sel_it in "${_sel_items[@]}"; do
    _sel_n=$((_sel_n + 1))
    if [[ "$_sel_it" =~ $_sel_re ]]; then
      case "${BASH_REMATCH[1]}" in proposed) _sel_proposed=$((_sel_proposed + 1)) ;; baseline) _sel_van=$((_sel_van + 1)) ;; esac
    else
      _sel_ok=0
    fi
  done
  if [ "$_sel_ok" != "1" ] || [ "$_sel_n" != "2" ] || [ "$_sel_proposed" != "1" ] || [ "$_sel_van" != "1" ]; then
    echo "FATAL: SELECTED must be 'proposed:<w>,baseline:<w>' (got '${SELECTED}'). Check that it was not cut at a comma of --export" >&2; exit 2
  fi
fi

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/aggregate_relax_novelty.py"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
case "$LAYERS" in additional) DEF_OUT="${RUN}/results/interim_additional" ;; main) DEF_OUT="${RUN}/results/interim_main" ;; both) DEF_OUT="${RUN}/results" ;; esac
OUT_DIR="${OUT_DIR:-$DEF_OUT}"
CMD=("$PY" "$SCRIPT" --layers "$LAYERS" --out-dir "$OUT_DIR" --procs "$PROCS")
if [ -n "$SELECTED" ]; then CMD+=(--selected "$SELECTED"); fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-} layers=${LAYERS} selected=${SELECTED:-<none>} out_dir=${OUT_DIR} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{relax_novelty_summary.json,relax_novelty_per_target.csv,relax_novelty_report.md,figs/relax_novelty_grid.png,figs/relax_novelty_grid.pdf}  ${RUN}/logs/agg_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] initial check: number of rows (additional 2,800 / main 700 / both 3,500), RELAX_DONE and NOVELTY_{PRE,POST}_DONE are present for all chunks. If even one is missing, nothing is written and it exits 2"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

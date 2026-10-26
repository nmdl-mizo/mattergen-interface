#!/bin/bash
# Metrics restricted to valid structures (G2, E): validation and production computation. CPU only (CPU partition). No MLIP is used.
# Description: see the experiment description (later differences take precedence).
# Usage (no --time on the CPU partition):
#   1) validation(F1〜F6): sbatch --export=ALL,MODE=validation sbatch_structure_filtered.sh   → results/structure_filtered/validation.json
#   2) production computation (refused at start unless validation.json is all_pass=true):
#        SELECTED=proposed:2.0,baseline:2.0 MODE=main sbatch sbatch_structure_filtered.sh   → results/structure_filtered/{summary.json,per_target.csv,report.md}
#      [Note] If SELECTED is written as --export=ALL,...,SELECTED=a,b it is cut at the comma (a baseline entry is lost). Pass it as an environment variable (the default --export=ALL of sbatch passes it on). If unspecified, MODE=main uses proposed:2.0,baseline:2.0.
#   The output is not overwritten (if it exists, Python raises FileExistsError and exits with 2). If the validation FAILs, the validation.json of the FAIL remains (exit 3).
# Environment variables: MODE (required validation|main) / SELECTED (only for main. Default proposed:2.0,baseline:2.0) / OUT_DIR (optional. Default results/structure_filtered) / DRYRUN (0|1)
# DRYRUN=1 MODE=validation|main bash sbatch_structure_filtered.sh : nothing is run or created; the command line and the write destinations are printed and it exits.
# Execution environment: the PY= and environment-variable lines of sbatch_a.sh (${GBREL_CODE}/gb_eval/experiments/relax_novelty/scripts/) are transcribed verbatim
#   (evaluation environment. numpy, ase, pymatgen).
# Exit codes: 0 = success / 2 = failure of the inputs or checks (including the refusal because the validation did not pass) / 3 = validation FAIL
#
#SBATCH --job-name=al_sfilt
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/sfilt_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/sfilt_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
MODE="${MODE:-}"
case "$MODE" in validation|main) ;; *) echo "FATAL: MODE must be validation|main (got '${MODE}')" >&2; exit 2 ;; esac
SELECTED="${SELECTED:-}"
if [ "$MODE" = "main" ] && [ -z "$SELECTED" ]; then SELECTED="proposed:2.0,baseline:2.0"; fi
if [ "$MODE" = "validation" ] && [ -n "$SELECTED" ]; then echo "FATAL: SELECTED is only for MODE=main" >&2; exit 2; fi
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

# ---- verbatim from the source (sbatch_a.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/aggregate_structure_filtered.py"
OUT_DIR="${OUT_DIR:-${RUN}/results/structure_filtered}"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
CMD=("$PY" "$SCRIPT" --mode "$MODE" --out-dir "$OUT_DIR" --procs "$PROCS")
if [ -n "$SELECTED" ]; then CMD+=(--selected "$SELECTED"); fi
VALIDATION_JSON="${OUT_DIR}/validation.json"

# <CLUSTER_MOUNT> on compute nodes and on the login node (for the confirmation of DRYRUN)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-} mode=${MODE} selected=${SELECTED:-<none>} out_dir=${OUT_DIR} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  _od="$(rp "$OUT_DIR")"
  if [ "$MODE" = "main" ]; then
    _wj="$(rp "$VALIDATION_JSON")"
    if [ -f "$_wj" ]; then _ap="$(grep -c '"all_pass": true' "$_wj" || true)"; echo "[DRYRUN] validation json: $_wj exists, all_pass=true lines=${_ap}"; else echo "[DRYRUN] validation json: ${VALIDATION_JSON} not found (exit 2 in production. Run MODE=validation first)"; fi
    echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{summary.json,per_target.csv,report.md}  ${RUN}/logs/sfilt_<jobname>_<jobid>.{out,err}"
    for _f in summary.json per_target.csv report.md; do if [ -e "${_od}/${_f}" ]; then echo "[DRYRUN][WARN] ${_f} already exists (FileExistsError -> exit 2 in production)"; fi; done
  else
    echo "[DRYRUN] WRITE TARGETS: ${VALIDATION_JSON}  ${RUN}/logs/sfilt_<jobname>_<jobid>.{out,err}"
    if [ -e "${_od}/validation.json" ]; then echo "[DRYRUN][WARN] validation.json already exists (FileExistsError -> exit 2 in production)"; fi
  fi
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
if [ "$MODE" = "main" ]; then
  [ -f "$VALIDATION_JSON" ] || { echo "FATAL: validation json not found: $VALIDATION_JSON (run MODE=validation first)" >&2; exit 2; }
  _ap="$(grep -c '"all_pass": true' "$VALIDATION_JSON" || true)"
  [ "${_ap:-0}" = "1" ] || { echo "FATAL: $VALIDATION_JSON is not all_pass=true (the production computation is refused. Python also checks it)" >&2; exit 2; }
fi
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

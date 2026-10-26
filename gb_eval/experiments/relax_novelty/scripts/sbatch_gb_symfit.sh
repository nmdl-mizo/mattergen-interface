#!/bin/bash
# Stages 0 and 1: equivalence of the two grain boundaries (residual of the 2_1 screw fit). CPU only. No MLIP is used.
# Plan: see the experiment description.
# [This file is not submitted by the author of this implementation] Submission is a separate step:
#   1) validation (W1 to W11. In the order of the plan. Exit non-zero at the first failure. One process is enough, so the CPU can be limited):
#        sbatch --cpus-per-task=2 --export=ALL,MODE=validation sbatch_gb_symfit.sh      → results/gb_equivalence/gb_symfit_validation.json
#   2) production computation (refused at start unless validation.json is all PASS):
#        sbatch --export=ALL,MODE=main sbatch_gb_symfit.sh                            → results/gb_equivalence/{gb_symfit_main.csv,gb_symfit_summary.md}
#   3) production computation of additional (refused at start unless validation.json is all PASS. OUT_DIR as default):
#        sbatch --export=ALL,MODE=additional sbatch_gb_symfit.sh                            → results/gb_equivalence/{gb_symfit_additional.csv,gb_symfit_summary_additional.md}
#   4) aggregation of G2 and E (standard library only; 2 CPUs are enough. LAYER required):
#        sbatch --cpus-per-task=2 --export=ALL,MODE=g2_and_e,LAYER=additional sbatch_gb_symfit.sh → results/gb_equivalence/gb_symfit_additional_g2_and_e.{md,json}
#        sbatch --cpus-per-task=2 --export=ALL,MODE=g2_and_e,LAYER=main OUT_PREFIX=<new prefix> sbatch_gb_symfit.sh -> check of the known values of main (--check-known-main)
#      The CSV of g2_and_e (default OUT_DIR/gb_symfit_${LAYER}.csv) and OUT_PREFIX (default OUT_DIR/gb_symfit_${LAYER}_g2_and_e) can be replaced with the environment variables CSV and OUT_PREFIX.
#   Give no --time on the CPU partition. The output is not overwritten (if it exists, Python raises FileExistsError and exits with 2).
#   If the validation FAILs, the validation.json of the FAIL remains. The author checks the content, isolates the cause, moves it away (mv) and reruns (thresholds, formulas and parameters are not changed).
# Environment variables: MODE (required validation|main|additional|g2_and_e) / LAYER (required for MODE=g2_and_e main|additional) / OUT_DIR (optional. Default results/gb_equivalence)
#           / CSV and OUT_PREFIX (optional only for MODE=g2_and_e) / DRYRUN (0|1)
# DRYRUN=1 MODE=validation|main|additional|g2_and_e bash sbatch_gb_symfit.sh : nothing is run or created; the command line, the resolution of the inputs and their counts, and the write destinations are printed and it exits.
# Execution environment: the PY=, PYTHONDONTWRITEBYTECODE and OMP lines of sbatch_a.sh (${GBREL_CODE}/gb_eval/experiments/relax_novelty/scripts/) are transcribed verbatim
#   (evaluation environment. numpy, scipy, ase).
# Exit codes: 0 = success / 2 = failure of the inputs or checks / 3 = validation FAIL (FAIL of the known-value check for MODE=g2_and_e LAYER=main)
#
#SBATCH --job-name=al_gbsym
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/gbsym_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/gbsym_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
MODE="${MODE:-}"
case "$MODE" in validation|main|additional|g2_and_e) ;; *) echo "FATAL: MODE must be validation|main|additional|g2_and_e (got '${MODE}')" >&2; exit 2 ;; esac
# LAYER: MODE=main -> main / MODE=additional -> additional / MODE=g2_and_e -> environment variable LAYER (main|additional required) / the validation uses the manifest of main
case "$MODE" in
  main) LAYER=main ;;
  additional) LAYER=additional ;;
  validation) LAYER=main ;;
  g2_and_e) case "${LAYER:-}" in main|additional) ;; *) echo "FATAL: MODE=g2_and_e requires LAYER=main|additional (got '${LAYER:-}')" >&2; exit 2 ;; esac ;;
esac

# ---- verbatim from the source (sbatch_a.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/gb_symfit.py"
OUT_DIR="${OUT_DIR:-${RUN}/results/gb_equivalence}"
if [ "$MODE" = "main" ] || [ "$MODE" = "additional" ]; then PROCS="${SLURM_CPUS_PER_TASK:-20}"; else PROCS=1; fi
MANIFEST="${RUN}/results/inputs_manifest_${LAYER}.csv"
if [ "$MODE" = "g2_and_e" ]; then
  CSV="${CSV:-${RUN}/results/gb_equivalence/gb_symfit_${LAYER}.csv}"
  OUT_PREFIX="${OUT_PREFIX:-${RUN}/results/gb_equivalence/gb_symfit_${LAYER}_g2_and_e}"
  G2_AND_E_SCRIPT="${RUN}/scripts/summarize_g2_and_e.py"
  CMD=("$PY" "$G2_AND_E_SCRIPT" --layer "$LAYER" --csv "$CSV" --manifest "$MANIFEST" --out-prefix "$OUT_PREFIX")
  if [ "$LAYER" = "main" ]; then CMD+=(--check-known-main); fi
else
  CMD=("$PY" "$SCRIPT" --mode "$MODE" --out-dir "$OUT_DIR" --procs "$PROCS")
fi

# <CLUSTER_MOUNT> on compute nodes and on the login node (for the confirmation of DRYRUN)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
TRAIN_MANIFEST="${RUN}/results/train_ref_manifest.csv"
VALIDATION_JSON="${OUT_DIR}/gb_symfit_validation.json"

echo "host=$(hostname) job=${SLURM_JOB_ID:-} mode=${MODE} layer=${LAYER} out_dir=${OUT_DIR} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  _m="$(rp "$MANIFEST")"; _t="$(rp "$TRAIN_MANIFEST")"
  if [ -f "$_m" ]; then echo "[DRYRUN] manifest: ${MANIFEST} -> $_m rows(lines-1)=$(( $(wc -l < "$_m") - 1 ))"; else echo "[DRYRUN][WARN] manifest not found: ${MANIFEST}"; fi
  if [ "$MODE" != "g2_and_e" ]; then
    if [ -f "$_t" ]; then echo "[DRYRUN] train manifest: ${TRAIN_MANIFEST} -> $_t rows(lines-1)=$(( $(wc -l < "$_t") - 1 ))"; else echo "[DRYRUN][WARN] train manifest not found: ${TRAIN_MANIFEST}"; fi
    # target of the relaxation check: validation/main = w2.0 of the main layer only (arm 2 x chunk 14 = 28), additional = w1.0 to 4.0 (arm 2 x w 4 x chunk 14 = 112)
    if [ "$MODE" = "additional" ]; then _ws="1.0 2.0 3.0 4.0"; _exp=112; else _ws="2.0"; _exp=28; fi
    _done=0; _tot=0
    for _arm in proposed baseline; do
      for _w in $_ws; do
        for _c in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
          _tot=$((_tot + 1))
          _cn="$(printf 'chunk%02d' "$_c")"
          _d="$(rp "${RUN}/results/relax/${LAYER}/${_arm}_w${_w}/${_cn}")"
          if [ -f "${_d}/RELAX_DONE" ] && [ -f "${_d}/relaxed.extxyz" ] && [ -f "${_d}/relax_rows.csv" ]; then _done=$((_done + 1)); fi
        done
      done
    done
    echo "[DRYRUN] relax chunks with RELAX_DONE+relaxed.extxyz+relax_rows.csv: ${_done}/${_tot} (expected ${_exp}/${_exp})"
  fi
  _od="$(rp "$OUT_DIR")"
  if [ "$MODE" = "main" ] || [ "$MODE" = "additional" ]; then
    if [ "$MODE" = "additional" ]; then _cn_out=gb_symfit_additional.csv; _sn_out=gb_symfit_summary_additional.md; _exp_rows="5,600 rows = 2,800 x {pre,post}"; else _cn_out=gb_symfit_main.csv; _sn_out=gb_symfit_summary.md; _exp_rows="1,400 rows = 700 x {pre,post}"; fi
    _wj="$(rp "$VALIDATION_JSON")"
    if [ -f "$_wj" ]; then _ap="$(grep -c '"all_pass": true' "$_wj" || true)"; echo "[DRYRUN] validation json: $_wj exists, all_pass=true lines=${_ap}"; else echo "[DRYRUN] validation json: ${VALIDATION_JSON} not found (exit 2 in production. Run MODE=validation first)"; fi
    echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/${_cn_out} (expected ${_exp_rows})  ${OUT_DIR}/${_sn_out}  ${RUN}/logs/gbsym_<jobname>_<jobid>.{out,err}"
    for _f in "$_cn_out" "$_sn_out"; do if [ -e "${_od}/${_f}" ]; then echo "[DRYRUN][WARN] ${_f} already exists (FileExistsError -> exit 2 in production)"; fi; done
  elif [ "$MODE" = "g2_and_e" ]; then
    _c="$(rp "$CSV")"
    if [ -f "$_c" ]; then echo "[DRYRUN] csv: ${CSV} -> $_c rows(lines-1)=$(( $(wc -l < "$_c") - 1 ))"; else echo "[DRYRUN][WARN] csv not found: ${CSV}"; fi
    echo "[DRYRUN] WRITE TARGETS: ${OUT_PREFIX}.md  ${OUT_PREFIX}.json  ${RUN}/logs/gbsym_<jobname>_<jobid>.{out,err}"
    for _f in "${OUT_PREFIX}.md" "${OUT_PREFIX}.json"; do if [ -e "$(rp "$_f")" ]; then echo "[DRYRUN][WARN] ${_f} already exists (FileExistsError -> exit 2 in production)"; fi; done
  else
    echo "[DRYRUN] WRITE TARGETS: ${VALIDATION_JSON}  ${RUN}/logs/gbsym_<jobname>_<jobid>.{out,err}"
    if [ -e "${_od}/gb_symfit_validation.json" ]; then echo "[DRYRUN][WARN] gb_symfit_validation.json already exists (FileExistsError -> exit 2 in production)"; fi
  fi
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
if [ "$MODE" = "g2_and_e" ]; then
  [ -f "$G2_AND_E_SCRIPT" ] || { echo "FATAL: script not found: $G2_AND_E_SCRIPT" >&2; exit 2; }
  [ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 2; }
  [ -f "$CSV" ] || { echo "FATAL: csv not found: $CSV" >&2; exit 2; }
else
  [ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
  [ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 2; }
  [ -f "$TRAIN_MANIFEST" ] || { echo "FATAL: train manifest not found: $TRAIN_MANIFEST" >&2; exit 2; }
fi
if [ "$MODE" = "main" ] || [ "$MODE" = "additional" ]; then
  [ -f "$VALIDATION_JSON" ] || { echo "FATAL: validation json not found: $VALIDATION_JSON (run MODE=validation first)" >&2; exit 2; }
  _ap="$(grep -c '"all_pass": true' "$VALIDATION_JSON" || true)"
  [ "${_ap:-0}" = "1" ] || { echo "FATAL: $VALIDATION_JSON is not all_pass=true (the production computation is refused. Python also checks it)" >&2; exit 2; }
fi
cd "${RUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

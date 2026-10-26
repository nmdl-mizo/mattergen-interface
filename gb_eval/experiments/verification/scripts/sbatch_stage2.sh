#!/bin/bash
# Stage 2 (equivalence decision of the two grain boundaries in a local window around the grain boundary): CPU only. No MLIP is used.
# Plan: see the experiment description.
# Submitted with sbatch on the cluster login node (the control node has no sbatch). CPU partition. No --time. Outputs are not overwritten (Python exits non-zero if they already exist).
#   MODE (required) and submission examples (pass --job-name as al_s2_<mode>. Chain dependencies with --dependency=afterok:<preceding stage>):
#     pack_vrun     : post and two_gb structures of the layers of the VRUN family (main, baseline_regen1 to 3, proposed_regen1, proposed_regen2) into inputs/<layer>.extxyz
#     pack_run      : layer of the RUN family (additional) into inputs/additional.extxyz (may run in parallel with pack_vrun)
#     validation       : W1 to W6 -> results/gb_stage2/gb_stage2_validation.json (run is refused unless all_pass). afterok:pack_vrun
#     r0            : stage R0 -> results/gb_stage2/dry_run_counts.md (exit 4 if a stop condition is hit). afterok:validation,pack_run
#     self_check_gate : verdict validation of the gate function -> results/gb_stage2/self_check_gate.json (no dependency)
#     ---- up to here is the implementation stage. The following are submitted by the parent (job runner) after approval ----
#     run           : stage R. Three times with h=4.0 (main), 3.0 and 6.0 -> gb_stage2_<layer>{,_h3,_h6}.csv (7 layers)
#     aggregate     : aggregation -> gate_inputs.json, gb_stage2_summary.md. afterok:run
#   Example: ssh <LOGIN_NODE> 'cd ${GBREL_CODE}/gb_eval/experiments/verification/scripts && sbatch --job-name=al_s2_pack_vrun --export=ALL,MODE=pack_vrun sbatch_stage2.sh'
# Completion: at the end of each MODE, ${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py write-done-marker (n>0 guard; counts directly under --dir) -> prints "done rc=0 mode=<MODE>".
#   Acceptance: the log contains 'done rc=0'. The DONE marker is DONE_<mode> in each output dir.
# Environment variables: MODE (required) / DRYRUN (0|1)
# DRYRUN=1 MODE=<MODE> bash sbatch_stage2.sh : executes and creates nothing; prints the command line, the resolved inputs with counts, and the write destinations, then exits.
# Execution environment: the PY=, PYTHONDONTWRITEBYTECODE and OMP lines of sbatch_verify.sh (same directory) copied verbatim (conda mace-mh; numpy, scipy, ase).
# Exit code: 0 = success / 2 = failure of the inputs or checks / 3 = validation or self_check FAIL / 4 = R0 stop condition
# The model-specification check is not called.
#
#SBATCH --job-name=al_stage2
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/stage2_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage2_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
MODE="${MODE:-}"
case "$MODE" in pack_vrun|pack_run|validation|r0|run|aggregate|self_check_gate) ;; *) echo "FATAL: MODE must be pack_vrun|pack_run|validation|r0|run|aggregate|self_check_gate (got '${MODE}')" >&2; exit 2 ;; esac

# ---- verbatim from the source (sbatch_verify.sh:48-50) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

VRUN="${GBREL_WORK}/workspace/experiments/verification"
RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SC="${VRUN}/scripts"
OUT="${VRUN}/results/gb_stage2"
IN="${OUT}/inputs"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
RG="${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py"
VLAYERS="main baseline_regen1 baseline_regen2 baseline_regen3 proposed_regen1 proposed_regen2"
ALL_LAYERS="main additional baseline_regen1 baseline_regen2 baseline_regen3 proposed_regen1 proposed_regen2"
VLAYERS_CSV="main,baseline_regen1,baseline_regen2,baseline_regen3,proposed_regen1,proposed_regen2"

# Command sequence for each MODE (CMDS may be several), completion marker (dir, glob list, marker), write destinations
CMDS=()
M_DIR="$OUT"; M_MARK="DONE_${MODE}"; M_GLOBS=(); W_TARGETS=""
case "$MODE" in
  pack_vrun)     CMDS=("$PY ${SC}/gb_stage2_pack.py --source vrun --layers ${VLAYERS_CSV} --out-dir ${IN}")
                 M_DIR="$IN"; for L in $VLAYERS; do M_GLOBS+=("${L}.extxyz"); done
                 for L in $VLAYERS; do W_TARGETS+="${IN}/${L}.extxyz "; done ;;
  pack_run)      CMDS=("$PY ${SC}/gb_stage2_pack.py --source run --layers additional --out-dir ${IN}")
                 M_DIR="$IN"; M_GLOBS=("additional.extxyz"); W_TARGETS="${IN}/additional.extxyz" ;;
  validation)       CMDS=("$PY ${SC}/gb_stage2_local.py --mode validation")
                 M_GLOBS=("gb_stage2_validation.json"); W_TARGETS="${OUT}/gb_stage2_validation.json" ;;
  r0)            CMDS=("$PY ${SC}/gb_stage2_local.py --mode r0 --procs ${PROCS}")
                 M_GLOBS=("dry_run_counts.md"); W_TARGETS="${OUT}/dry_run_counts.md" ;;
  run)           CMDS=("$PY ${SC}/gb_stage2_local.py --mode run --h 4.0 --procs ${PROCS}"
                       "$PY ${SC}/gb_stage2_local.py --mode run --h 3.0 --procs ${PROCS}"
                       "$PY ${SC}/gb_stage2_local.py --mode run --h 6.0 --procs ${PROCS}")
                 M_GLOBS=("gb_stage2_*.csv"); W_TARGETS="${OUT}/gb_stage2_<layer>{,_h3,_h6}.csv (7 layers x 3)" ;;
  aggregate)     CMDS=("$PY ${SC}/gb_stage2_local.py --mode aggregate")
                 M_GLOBS=("stage2_check_inputs.json"); W_TARGETS="${OUT}/stage2_check_inputs.json ${OUT}/gb_stage2_summary.md" ;;
  self_check_gate) CMDS=("$PY ${SC}/gb_stage2_local.py --mode self_check-gate")
                 M_GLOBS=("self_check_gate.json"); W_TARGETS="${OUT}/self_check_gate.json" ;;
esac

# <CLUSTER_MOUNT> on compute nodes and on the login node (for the confirmation of DRYRUN)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-} mode=${MODE} procs=${PROCS}"
if [ "$DRYRUN" = "1" ]; then
  for c in "${CMDS[@]}"; do echo "[DRYRUN] python command: ${c}"; done
  case "$MODE" in
    pack_vrun)
      for L in $VLAYERS; do
        _c="$(rp "${VRUN}/results/gb_equivalence/gb_symfit_${L}.csv")"
        if [ -f "$_c" ]; then echo "[DRYRUN] ${L}: stage1 csv ${_c} rows(lines-1)=$(( $(wc -l < "$_c") - 1 )) approximate number of post and two_gb rows (grep)=$(grep -c ',post,.*,two_gb,' "$_c" || true)"; else echo "[DRYRUN][WARN] stage1 csv not found: ${_c}"; fi
      done ;;
    pack_run)
      _c="$(rp "${RUN}/results/gb_equivalence/gb_symfit_additional.csv")"
      if [ -f "$_c" ]; then echo "[DRYRUN] additional: stage1 csv ${_c} rows(lines-1)=$(( $(wc -l < "$_c") - 1 )) (expected 5600) approximate number of post and two_gb rows (grep)=$(grep -c ',post,.*,two_gb,' "$_c" || true) (expected 1,618 = baseline 40+52+63+68 + proposed 350+350+350+345)"; else echo "[DRYRUN][WARN] stage1 csv not found: ${_c}"; fi ;;
    validation|r0|run|aggregate)
      for L in $ALL_LAYERS; do
        _p="$(rp "${IN}/${L}.extxyz")"
        if [ -f "$_p" ]; then echo "[DRYRUN] pack ${L}: ${_p} frames (number of 'Lattice=' lines)=$(grep -c 'Lattice=' "$_p" || true)"; else echo "[DRYRUN][WARN] pack not found: ${_p} (pack first)"; fi
      done
      if [ "$MODE" = "run" ]; then
        _wj="$(rp "${OUT}/gb_stage2_validation.json")"
        if [ -f "$_wj" ]; then echo "[DRYRUN] validation json: $_wj all_pass=true lines=$(grep -c '"all_pass": true' "$_wj" || true) (run is refused unless it is 1)"; else echo "[DRYRUN] validation json not found (exit 2 in production)"; fi
      fi ;;
  esac
  echo "[DRYRUN] WRITE TARGETS: ${W_TARGETS}  ${M_DIR}/${M_MARK}  ${VRUN}/logs/stage2_<jobname>_<jobid>.{out,err}"
  for _f in ${W_TARGETS}; do if [ -e "$(rp "$_f")" ]; then echo "[DRYRUN][WARN] ${_f} already exists (FileExistsError -> exit 2 in production)"; fi; done
  if [ -e "$(rp "${M_DIR}/${M_MARK}")" ]; then echo "[DRYRUN][WARN] marker ${M_DIR}/${M_MARK} already exists"; fi
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "${SC}/gb_stage2_local.py" ] || { echo "FATAL: script not found: ${SC}/gb_stage2_local.py" >&2; exit 2; }
if [ -e "${M_DIR}/${M_MARK}" ]; then echo "FATAL: marker ${M_DIR}/${M_MARK} already exists (outputs are not overwritten)" >&2; exit 2; fi
mkdir -p "$M_DIR"
cd "$SC"
echo "start: $(date -u)"
for c in "${CMDS[@]}"; do
  echo "+ ${c}"
  # shellcheck disable=SC2086
  $c
done
echo "end: $(date -u)"
# Check that the number of products n>0 before the completion marker (the check and the marker writing are inseparable)
for g in "${M_GLOBS[@]}"; do
  "$PY" "$RG" write-done-marker --dir "$M_DIR" --glob "$g" --min-count 1 --marker "$M_MARK"
done
echo "done rc=0 mode=${MODE}"

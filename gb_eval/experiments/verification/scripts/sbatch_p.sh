#!/bin/bash
# Manifest of the layer (inputs_manifest_<layer>.csv, array_map_<layer>.csv) and the identity check of the fixed atoms of proposed (P2-A/B). CPU only.
# Original: ${GBREL_CODE}/gb_eval/experiments/relax_novelty/scripts/sbatch_p.sh (RUN is not changed). Only the layer (LAYER), the output destination (VRUN), the entrance check and the completion marker were changed.
#   The reference to the training set (train_ref_manifest) uses the existing one of RUN read only (build_manifests.py only checks it; not rebuilt for each layer).
# [Not submitted by the author] Submission is by the job runner via submit_layer_chain.sh:
#   sbatch --job-name=al_manifest_<layer> --dependency=afterok:<eval> --export=ALL,LAYER=<layer> sbatch_p.sh      (no --time on the CPU partition)
# DRYRUN=1 LAYER=<layer> bash sbatch_p.sh : executes and creates nothing; prints the resolved values, the command line and the write destinations, then exits.
# Execution environment: the VENV_PYTHON, SCM_HOST_Z, SCM_SOLUTE_ZS, PYTHONPATH and PYTHONDONTWRITEBYTECODE lines of sbatch_p.sh of RUN copied verbatim (mattergen venv; used by fixed_identity).
# Completion: runner_guards write-done-marker (counts inputs_manifest_<layer>.csv directly under results/; n>0 guard) -> results/P_DONE_<layer>.
# Exit code: 0 = success / 2 = failure of the inputs or checks / 3 = P2-A or P2-B FAIL on real data (stops without changing the thresholds or formulas)
#
#SBATCH --job-name=al_manifest
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/manifest_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/manifest_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-}"
SC="${GBREL_CODE}/gb_eval/experiments/verification/scripts"
[ -d "$SC" ] || SC="/mn""t${SC}"
source "${SC}/layer_lib.sh"
layer_require "$LAYER"
export PYTHONDONTWRITEBYTECODE=1

# ---- verbatim from the source (sbatch_p.sh of RUN) ----
VENV_PYTHON="${GEN_PY}"
export SCM_HOST_Z=13
export SCM_SOLUTE_ZS=
export PYTHONPATH="${GBREL_CODE}"

RUN="${VRUN}"
SCRIPT="${RUN}/scripts/build_manifests.py"
RESULTS_DIR="${LAYER_WORK_ROOT}/results"
VERIFICATION_DIR="${RUN}/verification"
CMD=("$VENV_PYTHON" "$SCRIPT" --layers "$LAYER" --results-dir "$RESULTS_DIR" --verification-dir "$VERIFICATION_DIR")
RG="${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py"

echo "host: $(hostname)"
echo "DRYRUN=${DRYRUN} LAYER=${LAYER} arm=${LAYER_ARM} SCM_HOST_Z=${SCM_HOST_Z} SCM_SOLUTE_ZS='${SCM_SOLUTE_ZS}' PYTHONPATH=${PYTHONPATH}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] entrance gate: ${VENV_PYTHON} ${SC}/stage_gate.py --layer ${LAYER} --stage eval (ALL_DONE of the preceding eval stage, g63v PASS, 350 rows of gamma_pre)"
  echo "[DRYRUN] marker: ${VENV_PYTHON} ${RG} write-done-marker --dir ${RESULTS_DIR} --glob inputs_manifest_${LAYER}.csv --min-count 1 --marker P_DONE_${LAYER}"
  echo "[DRYRUN] WRITE TARGETS: ${RESULTS_DIR}/inputs_manifest_${LAYER}.csv  ${RESULTS_DIR}/array_map_${LAYER}.csv  ${VERIFICATION_DIR}/P_summary_${LAYER}.json  ${RESULTS_DIR}/P_DONE_${LAYER}"
  if [ "$LAYER_ARM" = "proposed" ]; then echo "[DRYRUN]   ${VERIFICATION_DIR}/proposed_fixed_identity_${LAYER}.json"; fi
  for _f in "${RESULTS_DIR}/inputs_manifest_${LAYER}.csv" "${RESULTS_DIR}/array_map_${LAYER}.csv"; do
    _r="$_f"; [ -e "$_r" ] || _r="/mn""t${_f}"; if [ -e "$_r" ]; then echo "[DRYRUN][WARN] ${_f} already exists (exit 2 in production; not overwritten)"; fi
  done
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$VENV_PYTHON" ] || { echo "FATAL: venv python not executable: $VENV_PYTHON" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
"$VENV_PYTHON" "${SC}/stage_gate.py" --layer "$LAYER" --stage eval   # Entrance check (products of the preceding stage: 350)
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"
"$VENV_PYTHON" "$RG" write-done-marker --dir "$RESULTS_DIR" --glob "inputs_manifest_${LAYER}.csv" --min-count 1 --marker "P_DONE_${LAYER}"
"$VENV_PYTHON" "${SC}/stage_gate.py" --layer "$LAYER" --stage manifest   # Exit check
echo "done rc=0 layer=${LAYER}"

#!/bin/bash
# Stage P: list of inputs, array mapping, list of train references, identity check of the fixed atoms of proposed (P2-A/B), inputs for R3. CPU only (no GPU is taken).
#   Submission: sbatch --export=ALL,LAYERS=additional sbatch_p.sh      (LAYERS=additional|main|both. Default additional. No --time on the CPU partition)
#   DRYRUN=1 bash sbatch_p.sh : nothing is run or created; the resolved values, the command line and the write destinations are printed and it exits.
#   Switching the output destinations for development: RESULTS_DIR=<dir> VERIFICATION_DIR=<dir> (if unset, results/ and verification/ of this experiment).
# Execution environment: the lines VENV_PYTHON, SCM_HOST_Z, SCM_SOLUTE_ZS, PYTHONPATH and PYTHONDONTWRITEBYTECODE of sbatch_gen_proposed.sh (${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/)
#   are transcribed verbatim.
# Exit codes: 0 = success / 2 = failure of the inputs or checks / 3 = P2-A or P2-B FAIL on real data (stop without changing the threshold or the formula)
#
#SBATCH --job-name=al_rn_p
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/p_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/p_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYERS_ARG="${LAYERS:-additional}"
case "$LAYERS_ARG" in additional|main|both) ;; *) echo "FATAL: LAYERS must be additional|main|both (got '${LAYERS_ARG}')" >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1

# ---- verbatim from the source (sbatch_gen_proposed.sh) ----
VENV_PYTHON="${GEN_PY}"
export SCM_HOST_Z=13
export SCM_SOLUTE_ZS=
export PYTHONPATH="${GBREL_CODE}"

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
SCRIPT="${RUN}/scripts/build_manifests.py"
RESULTS_DIR="${RESULTS_DIR:-${RUN}/results}"
VERIFICATION_DIR="${VERIFICATION_DIR:-${RUN}/verification}"
CMD=("$VENV_PYTHON" "$SCRIPT" --layers "$LAYERS_ARG" --results-dir "$RESULTS_DIR" --verification-dir "$VERIFICATION_DIR")

echo "host: $(hostname)"
echo "DRYRUN=${DRYRUN} LAYERS=${LAYERS_ARG} SCM_HOST_Z=${SCM_HOST_Z} SCM_SOLUTE_ZS='${SCM_SOLUTE_ZS}' PYTHONPATH=${PYTHONPATH}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] WRITE TARGETS (under ${RUN}/ unless overridden):"
  for L in $( [ "$LAYERS_ARG" = both ] && echo "additional main" || echo "$LAYERS_ARG" ); do
    echo "[DRYRUN]   ${RESULTS_DIR}/inputs_manifest_${L}.csv  ${RESULTS_DIR}/array_map_${L}.csv  ${VERIFICATION_DIR}/P_summary_${L}.json"
    if [ "$L" = additional ]; then echo "[DRYRUN]   ${RESULTS_DIR}/r1_additional_array_spec.txt  ${VERIFICATION_DIR}/proposed_fixed_identity.json"; else echo "[DRYRUN]   ${VERIFICATION_DIR}/proposed_fixed_identity_main.json"; fi
  done
  echo "[DRYRUN]   ${RESULTS_DIR}/train_ref_manifest.csv (if absent)  ${VERIFICATION_DIR}/r3_inputs.csv (if absent)  ${RUN}/logs/p_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$VENV_PYTHON" ] || { echo "FATAL: venv python not executable: $VENV_PYTHON" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

#!/bin/bash
# Aggregation of the neighbourhood achievement rate of the unconditional generation (w=0) of proposed (w0_baseline.py). CPU only. No MLIP is used. No decision (gate) is made (descriptive).
# Plan: see the experiment description.
# [Not submitted by the author] Submission is by the job runner. Run only one, after the novelty of the chain (submit_layer_chain.sh proposed_w0 --submit) has completed (afterok):
#   sbatch --job-name=al_w0_baseline --dependency=afterok:<novelty>:<theta> --export=ALL,LAYER=proposed_w0 sbatch_w0_baseline.sh   (no --time on the CPU partition)
#   If theta (-> g2_and_e) is also awaited, w0_gamma_summary.json carries the layer aggregation of g2_and_e (counts of G2 and E). If only novelty is awaited, there is no g2_and_e and that field becomes null (not an error).
# Entrance check: relax and novelty of stage_gate.py (per layer 14 x 25 = 350 structures; non-zero exit if short). Completion marker: runner_guards write-done-marker (base_rate.csv n>0).
# Execution environment: the PY= (conda mace-mh; numpy, matplotlib, ase) and PYTHONDONTWRITEBYTECODE lines of sbatch_n.sh / sbatch_verify.sh copied verbatim.
# Environment variables: LAYER (default proposed_w0; only a layer with one proposed arm and ws=[0.0] is allowed) / DRYRUN (0|1)
# DRYRUN=1 bash sbatch_w0_baseline.sh : executes and creates nothing; prints the resolved inputs, counts, command line and write destinations, then exits (python is not started on the control node).
# Exit code: 0 = success / 2 = failure of the inputs or checks / 3 = the train counts disagree with the known values (the output is kept). Acceptance: the log contains 'done rc=0'.
#
#SBATCH --job-name=al_w0_baseline
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/w0baseline_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/w0baseline_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-proposed_w0}"
SC="${GBREL_CODE}/gb_eval/experiments/verification/scripts"
[ -d "$SC" ] || SC="/mn""t${SC}"
source "${SC}/layer_lib.sh"
layer_require "$LAYER"
[ "$LAYER_ARM" = "proposed" ] || { echo "FATAL: layer ${LAYER} is not a proposed layer (arm=${LAYER_ARM}; C1)" >&2; exit 2; }
[ "$LAYER_W" = "0.0" ] || { echo "FATAL: layer ${LAYER} has w=${LAYER_W}, but this aggregate is for the w=0.0 control only (C1)" >&2; exit 2; }

# ---- verbatim from the source (sbatch_n.sh / sbatch_verify.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"   # per-target CSV of the main experiment and train_ref_manifest (read only)
SCRIPT="${VRUN}/scripts/w0_baseline.py"
OUT_DIR="${VRUN}/results/w0_baseline"
RG="${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py"
ARMW="${LAYER_ARM}_w${LAYER_W}"
MANIFEST="${LAYER_WORK_ROOT}/results/inputs_manifest_${LAYER}.csv"
RELAX_ROOT="${LAYER_WORK_ROOT}/results/relax/${LAYER}/${ARMW}"
G2_AND_E_JSON="${VRUN}/results/gb_equivalence/gb_symfit_${LAYER}_g2_and_e.json"
CMD=("$PY" "$SCRIPT" --layer "$LAYER" --out-dir "$OUT_DIR")

# <CLUSTER_MOUNT> on compute nodes and on the login node (for the confirmation of DRYRUN)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-} layer=${LAYER} arm=${LAYER_ARM} W=${LAYER_W} armw=${ARMW}"
if [ "$DRYRUN" = "1" ]; then
  printf '[DRYRUN] self_check command: '; printf '%q ' "$PY" "$SCRIPT" --self_check; printf '\n'
  printf '[DRYRUN] python command: '; printf '%q ' "${CMD[@]}"; printf '\n'
  echo "[DRYRUN] entrance gates: ${PY} ${SC}/stage_gate.py --layer ${LAYER} --stage relax ; --stage novelty (per layer 14 x 25 = 350 structures)"
  _m="$(rp "$MANIFEST")"
  if [ -f "$_m" ]; then echo "[DRYRUN] manifest: ${MANIFEST} rows(lines-1)=$(( $(wc -l < "$_m") - 1 )) (expected 350)"; else echo "[DRYRUN][WARN] manifest not found (FATAL in production. Made by stage P of the chain): ${MANIFEST}"; fi
  _done=0; for _c in 0 1 2 3 4 5 6 7 8 9 10 11 12 13; do
    _d="$(rp "${RELAX_ROOT}/$(printf 'chunk%02d' "$_c")")"; if [ -f "${_d}/RELAX_DONE" ] && [ -f "${_d}/relax_rows.csv" ]; then _done=$((_done + 1)); fi
  done
  echo "[DRYRUN] relax chunks(RELAX_DONE+relax_rows.csv) under ${RELAX_ROOT}: ${_done}/14 (expected 14/14)"
  _p="$(rp "${RUN}/results/relax_novelty_per_target.csv")"
  if [ -f "$_p" ]; then echo "[DRYRUN] main per-target: ${RUN}/results/relax_novelty_per_target.csv proposed w=2.0 rows=$(grep -c '^main,proposed,2.0,' "$_p" || true) (expected 7)"; else echo "[DRYRUN][WARN] per-target csv not found: ${RUN}/results/relax_novelty_per_target.csv"; fi
  _t="$(rp "${RUN}/results/train_ref_manifest.csv")"
  if [ -f "$_t" ]; then echo "[DRYRUN] train_ref_manifest: ${RUN}/results/train_ref_manifest.csv rows(lines-1)=$(( $(wc -l < "$_t") - 1 )) (expected 4419)"; else echo "[DRYRUN][WARN] train_ref_manifest not found: ${RUN}/results/train_ref_manifest.csv"; fi
  _g="$(rp "$G2_AND_E_JSON")"
  if [ -f "$_g" ]; then echo "[DRYRUN] g2_and_e json: ${G2_AND_E_JSON} (present. The counts of G2 and E go into the summary)"; else echo "[DRYRUN] g2_and_e json not found: ${G2_AND_E_JSON} (if absent, g2_and_e_layer of w0_gamma_summary.json is null)"; fi
  echo "[DRYRUN] marker: ${PY} ${RG} write-done-marker --dir ${OUT_DIR} --glob base_rate.csv --min-count 1 --marker DONE_w0_baseline"
  echo "[DRYRUN] WRITE TARGETS: ${OUT_DIR}/{base_rate.csv,w0_gamma_summary.json,fig_w0_vs_w2.png}  ${OUT_DIR}/DONE_w0_baseline  ${VRUN}/logs/w0baseline_<jobname>_<jobid>.{out,err}"
  for _f in base_rate.csv w0_gamma_summary.json fig_w0_vs_w2.png; do if [ -e "$(rp "${OUT_DIR}/${_f}")" ]; then echo "[DRYRUN][WARN] ${OUT_DIR}/${_f} already exists (exit 2 in production; not overwritten)"; fi; done
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
for _st in relax novelty; do "$PY" "${SC}/stage_gate.py" --layer "$LAYER" --stage "$_st"; done   # Entrance check (products of the preceding stage: 350)
mkdir -p "$OUT_DIR"
cd "$SC"
echo "self_check: $(date -u)"
"$PY" "$SCRIPT" --self_check   # Check of the pure functions (the denominator stays 350 even for degenerate inputs, etc.). Stops here if it fails
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"
"$PY" "$RG" write-done-marker --dir "$OUT_DIR" --glob "base_rate.csv" --min-count 1 --marker "DONE_w0_baseline"
echo "done rc=0 mode=w0_baseline layer=${LAYER}"

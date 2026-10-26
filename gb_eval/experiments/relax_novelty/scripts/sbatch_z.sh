#!/bin/bash
# Noise-floor stage: noise floor of gamma before relaxation. gamma_pre is recomputed twice on the same node as the original evaluation (1 job = 1 layer x 1 arm x reps 1 and 2). GPU (MLIP).
#   LAYER=additional: proposed w3.0 / baseline w3.0 (gen_eval)  LAYER=main: proposed w2.0 / baseline w2.0 (range_targets). In total 2 layers x 2 arms x 2 reps = 8 runs (4 jobs).
# [This file is not submitted by the author of stage P] Submission is a separate step. The host of the original evaluation job is specified with --nodelist (confirm the host with the DRYRUN below):
#   Z-additional (no dependency): sbatch --time=00:30:00 --nodelist=<host> --export=ALL,LAYER=additional,ARM=proposed sbatch_z.sh   (the same for baseline. The host is the host: line of the eval log)
#   Z-main (to be submitted after the completion of stage E of range_targets): sbatch --time=00:30:00 --nodelist=<host> --export=ALL,LAYER=main,ARM=proposed sbatch_z.sh
#   A --time is required on the GPU partition. Run the GPU pre-check and the job pre-check before submission. If that host cannot be used (down, or PENDING for 6 hours or more because it is full), report to the author and wait (do not substitute another node).
# At start this job compares hostname with the host of the original evaluation log and exits 2 if they differ (the noise floor of another node is not the "noise floor of the same node").
# How to take the host: for additional, the `host:` line of ${GBREL_WORK}/workspace/experiments/gen_eval/logs/eval_eval_<model>_w3.0_*.out,
#   for main, the `host:` line of ${GBREL_WORK}/workspace/experiments/range_targets/logs/eval_*<model>_w2.0_*.out (printed by the evaluation script sbatch_eval.sh:104).
#   If there are 0 logs, several logs, or two or more kinds of hosts, exit 2 (report to the author).
# Output: results/noise/<layer>_<model>_w<W>_rep<r>.csv (refused if it exists). The comparison is sbatch_z_compare.sh (noise_floor_compare.py).
# sbatch_eval.sh of the original evaluation is not used (the output destination is fixed and Step1 writes ALL_DONE into the generation dir). compute_gamma_pre_{proposed,baseline}.py are called directly.
# Environment variables: LAYER (required additional|main) / ARM (required proposed|baseline) / REPS (default "1 2") / OUT_DIR (default results/noise)
# DRYRUN=1 bash sbatch_z.sh : nothing is run or created; the resolution of the host, the command line and the write destinations are printed and it exits.
# Execution environment: the PY and PYTHONDONTWRITEBYTECODE lines of sbatch_stage_c_relax.sh (${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts/) are transcribed verbatim.
#   The scripts and arguments are from Step3 of sbatch_eval.sh (${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/) (TEMPLATE_A, ckpt_resolved and arm-label are also the same).
#
#SBATCH --job-name=al_rn_z
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --gres=shard:16   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/noise_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/noise_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
LAYER="${LAYER:-}"
case "$LAYER" in additional|main) ;; *) echo "FATAL: LAYER must be additional or main (got '${LAYER}')" >&2; exit 2 ;; esac
ARM="${ARM:-}"
case "$ARM" in proposed|baseline) ;; *) echo "FATAL: ARM must be proposed or baseline (got '${ARM}')" >&2; exit 2 ;; esac
REPS="${REPS:-1 2}"

# ---- verbatim from the source (sbatch_stage_c_relax.sh) ----
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1

RUN="${GBREL_WORK}/workspace/experiments/relax_novelty"
RUN_REF="${GBREL_WORK}/workspace/experiments/gen_eval"
RUN_MAIN="${GBREL_WORK}/workspace/experiments/range_targets"
TEMPLATE_A="54.15601152562611"   # TEMPLATE_A of sbatch_eval.sh (common to all Al CNID)
case "$LAYER" in
  additional) W="3.0"; RUN_ROOT="$RUN_REF"; LOG_DIR="${RUN_REF}/logs"; LOG_GLOB="eval_al_eval_${ARM}_w${W}_*.out" ;;
  main) W="2.0"; RUN_ROOT="$RUN_MAIN"; LOG_DIR="${RUN_MAIN}/logs"; LOG_GLOB="eval_*${ARM}_w${W}_*.out" ;;
esac
GEN_DIR="${RUN_ROOT}/results/gen/${ARM}/w${W}"
CKPT_RESOLVED="${RUN_ROOT}/results/ckpt_resolved_${ARM}.json"
ARM_LABEL="${ARM}_w${W}"
COMPUTE_PY="${RUN_REF}/scripts/compute_gamma_pre_${ARM}.py"
OUT_DIR="${OUT_DIR:-${RUN}/results/noise}"

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }

# ---- host of the original evaluation job (the `host:` line of the log). Exit 2 unless there is exactly 1 log and 1 host ----
LOGDIR_READ="$(rp "$LOG_DIR")"
shopt -s nullglob
LOGS=( "${LOGDIR_READ}"/${LOG_GLOB} )
shopt -u nullglob
EXPECT_HOST=""
if [ "${#LOGS[@]}" -ne 1 ]; then
  MSG="original eval log: expected exactly 1 match of ${LOG_DIR}/${LOG_GLOB}, found ${#LOGS[@]} (report to the author)"
  if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN][WARN] ${MSG}" >&2; else echo "FATAL: ${MSG}" >&2; exit 2; fi
else
  HOSTS="$(grep -h '^host:' "${LOGS[0]}" | sort -u || true)"
  NH="$(printf '%s\n' "$HOSTS" | grep -c . || true)"
  if [ "$NH" -ne 1 ]; then
    MSG="original eval log ${LOGS[0]}: expected exactly 1 distinct 'host:' line, found ${NH} (report to the author)"
    if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN][WARN] ${MSG}" >&2; else echo "FATAL: ${MSG}" >&2; exit 2; fi
  else
    EXPECT_HOST="${HOSTS#host: }"
  fi
fi

echo "host: $(hostname)"
echo "DRYRUN=${DRYRUN} LAYER=${LAYER} ARM=${ARM} W=${W} REPS='${REPS}' expected_host(original eval)='${EXPECT_HOST}' (log: ${LOGS[0]:-<none>})"
echo "GEN_DIR=${GEN_DIR} CKPT_RESOLVED=${CKPT_RESOLVED} OUT_DIR=${OUT_DIR}"

build_cmd() {  # $1=rep -> CMD array
  local out="${OUT_DIR}/${LAYER}_${ARM}_w${W}_rep${1}.csv"
  if [ "$ARM" = "proposed" ]; then
    CMD=("$PY" "$COMPUTE_PY" --output-root "$GEN_DIR" --arm-label "$ARM_LABEL" --w "$W" --ckpt-resolved "$CKPT_RESOLVED" --out "$out" --template-a "$TEMPLATE_A")
  else
    CMD=("$PY" "$COMPUTE_PY" --gen-root "$GEN_DIR" --arm-label "$ARM_LABEL" --w "$W" --ckpt-resolved "$CKPT_RESOLVED" --out "$out")
  fi
  OUT_CSV="$out"
}

if [ "$DRYRUN" = "1" ]; then
  [ -n "$EXPECT_HOST" ] && echo "[DRYRUN] to be given to sbatch: --nodelist=${EXPECT_HOST}"
  for r in $REPS; do build_cmd "$r"; printf "[DRYRUN] rep${r} python command: "; printf '%q ' "${CMD[@]}"; printf '\n'; echo "[DRYRUN]   WRITE: ${OUT_CSV}"; done
  if [ "$ARM" = "proposed" ] && [ ! -d "$(rp "${GEN_DIR}/chunk00/cif")" ]; then echo "[DRYRUN][WARN] ${GEN_DIR}/chunk00/cif not found (cif/ of proposed is made by the extract of stage E. FATAL in production)" >&2; fi
  echo "[DRYRUN]   + ${RUN}/logs/noise_<jobname>_<jobid>.{out,err}"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$COMPUTE_PY" ] || { echo "FATAL: compute script not found: $COMPUTE_PY" >&2; exit 2; }
[ -f "$CKPT_RESOLVED" ] || { echo "FATAL: ckpt_resolved not found: $CKPT_RESOLVED" >&2; exit 2; }
[ -d "$GEN_DIR" ] || { echo "FATAL: generated dir not found: $GEN_DIR" >&2; exit 2; }
if [ "$ARM" = "proposed" ] && [ ! -d "${GEN_DIR}/chunk00/cif" ]; then echo "FATAL: ${GEN_DIR}/chunk00/cif not found (cif/ of proposed has not been expanded. Run after the completion of stage E. Do not expand into an existing run)" >&2; exit 2; fi
[ -n "$EXPECT_HOST" ] || { echo "FATAL: expected host unknown" >&2; exit 2; }
[ "$(hostname)" = "$EXPECT_HOST" ] || { echo "FATAL: running on '$(hostname)' but the original evaluation ran on '${EXPECT_HOST}' (do not substitute another node. Resubmit with --nodelist=${EXPECT_HOST})" >&2; exit 2; }
mkdir -p "$OUT_DIR"
for r in $REPS; do build_cmd "$r"; [ ! -e "$OUT_CSV" ] || { echo "FATAL: output exists (no overwrite): $OUT_CSV" >&2; exit 2; }; done

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
"$PY" - <<'PYCHK'
import sys, torch
ok = torch.cuda.is_available(); n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0) if ok and n else None
print(f"torch.cuda.is_available={ok} device_count={n} device0={name}", flush=True)
if not ok:
    print("FATAL: no GPU available (an MLIP is not run on CPU)", file=sys.stderr); sys.exit(3)
PYCHK

echo "start: $(date -u)"
for r in $REPS; do
  build_cmd "$r"
  echo "=== rep${r}: $(date -u) -> ${OUT_CSV}"
  "${CMD[@]}"
done
echo "end: $(date -u)"

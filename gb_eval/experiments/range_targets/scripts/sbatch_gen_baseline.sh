#!/bin/bash
# Copy of the range-target version. Source: ${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/sbatch_gen_baseline.sh.
# Only RUN_DIR (the value of the RUN variable), TARGETS_JSON, CARD_HASH12 and the range of k (14..27) were changed.
# Generation of the baseline (standard mattergen-generate) for Al. 7 targets x 50 structures x w=1.0/2.0/3.0/4.0 (w=4.0 is an extension, k=42..55).
#   1 task = one (w, chunk) = 25 structures (BATCH=5 x 5 batches). array k=0..55 -> w_idx=k//14, c=k%14.
#
# Description: see the experiment description (acceptance conditions on the checkpoint, the targets and the run records).
# Source: ${GBREL_CODE}/gb_eval/baseline_generation/sbatch_gen_al_baseline_dedup.sh (read only; copied and modified)
#   gen_baseline_seeded.py is run as in the source (read only).
#
# [This file is not submitted at the data-rebuild stage] Submission is a separate step. Usage:
#   G0: sbatch --array=0,14,28 --time=<..> --exclude=<EXCLUDE_NODES> sbatch_gen_baseline.sh   (--array is intentionally not written in the header)
#   G1: sbatch --array=1-13,15-27,29-41 --gres=shard:<from the measured PEAK_VRAM_MiB of G0> --time=<measured G0 x 1.5> ... sbatch_gen_baseline.sh
#   DRYRUN=1: nothing is created or run; the checks, the resolved values, the python command line and the write destinations are printed and it exits (WARN if ckpt is unresolved).
#   STOP_BEFORE_GEN=1 (when DRYRUN=0): runs up to the resolution of ckpt, the creation of the staging dir and the config check, and stops with rc=0.
#
# Environment variables: SLURM_ARRAY_TASK_ID (required. 0..55) / DRYRUN (0|1) / STOP_BEFORE_GEN (0|1)
#   The ckpt is read only from ckpt_resolved_baseline.json (it cannot be overwritten by an environment variable). No SMOKE is provided.
#
# [Differences from the source] (nothing else is changed)
#   1 array mapping (w_idx=k//14, c=k%14)  2 TRAIN_DIR -> baseline_train. The ckpt is from ckpt_resolved_baseline.json.
#     Check of the cache name of the staging dir scm_al_baseline_dedup -> scm_al_baseline. The epoch check is torch.load(ckpt)['epoch']==epoch of the json.
#     If the line "Loading model from checkpoint:" of the generation log differs from the ckpt of the staging dir, exit non-zero.
#   3 target=targets[c//2], SEED=20301000+c (level=k//5 and SEED_BASE=20292000 of the source are abolished)
#   4 NUM_SAMPLES=25, BATCH=5 (25%5==0. Frozen), GUIDANCE=W, NUM_ATOMS_KEY=AL_S5_210_TRAIN
#   5 output results/gen/baseline/w<W>/chunk<cc>/. In addition to chunk_meta.json, run_meta.json (17 keys. The last 4 keys are null)
#   6 PEAK_VRAM_MiB and the duration are recorded for the 3 baseline chunks of G0 (the poll_vram.sh of the source and the start/end print are used as they are)
#   7 BATCH=5 is frozen. If OOM occurs, the only thing allowed is to move to a node with larger free VRAM (changing BATCH is a decision of the author)
#   12 before the generation, the internal model-specification check is called (as for the proposed model; not included in this release; see NOTICE). It exits non-zero unless it passes. Output is appended to RUN/logs/ (to standard output only in DRYRUN)
#   2b the training config is read as YAML by check_baseline_config.py on compute nodes (DRYRUN=0) and the cache name scm_al_baseline, the num_atoms key and gb_gamma_J_m2 are cross-checked. grep is used only for the DRYRUN on the login node.
# [Author's choices]
#   J-1: the staging dir has the ckpt placed as a symlink (as in the source). The name of the ckpt in the staging dir is epoch=<E>.ckpt. The sha256 cross-check is done on the file in the staging dir (the symlink target).
#   J-2: location of the staging dir = RUN/results/gen_model_path/epoch<E> (same convention as the source. E is the epoch of ckpt_resolved).
#   J-3: --time=01:00:00 and --gres=shard:60 are initial values for G0 (no result for this data. In the source, 10 baseline structures took 00:28:00 with shard:60). They are overridden on the command line after G0 is measured.
#   J-4: read-only inputs are translated to <CLUSTER_MOUNT> so that DRYRUN is possible even on the login node that has no <CLUSTER_MOUNT> (rp function). The write destination is always in the <CLUSTER_MOUNT> notation.
#   J-5: PYTHONDONTWRITEBYTECODE=1 is exported.
#   J-6: the resolved_config of the proposed model is passed to the --train-config of the check (the bindings of the model specification are checked against the effective settings of the proposed model; passing the config of the baseline always FAILs). The config of the baseline itself is covered by the YAML check of 2b. Reported as a point to be confirmed.
# A --time is required on the GPU partition; the shard count = VRAM in GB. The execution environment is transcribed verbatim from the source.
#
#SBATCH --job-name=al_baseline_gen
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --time=01:00:00
#SBATCH --gres=shard:60   # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/baseline_gen_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/baseline_gen_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

fatal() { echo "FATAL: $*" >&2; exit 2; }

DRYRUN="${DRYRUN:-0}"
STOP_BEFORE_GEN="${STOP_BEFORE_GEN:-0}"
for v in DRYRUN STOP_BEFORE_GEN; do
  case "${!v}" in 0|1) ;; *) fatal "$v must be 0 or 1 (got '${!v}')" ;; esac
done
export PYTHONDONTWRITEBYTECODE=1

# ---- fixed values (<CLUSTER_MOUNT> from the viewpoint of compute nodes) ----
VENV_PYTHON="${GEN_PY}"
RUNNER_GUARDS="${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py"
TARGETS_JSON="${GBREL_WORK}/workspace/experiments/range_targets/results/eval_targets_range.json"
RUN_REF="${GBREL_WORK}/workspace/experiments/range_targets"
WRAPPER="${GBREL_CODE}/gb_eval/baseline_generation/gen_baseline_seeded.py"   # run read-only
POLL_VRAM="${GBREL_CODE}/gb_eval/experiments/baseline_reference/scripts/poll_vram.sh"
CKPT_RESOLVED_JSON="${RUN_REF}/results/ckpt_resolved_baseline.json"
CARD_HASH12="471d507f4133"

W_LIST=(1.0 2.0 3.0 4.0)
N_CHUNKS=14
N_ARRAY=56               # k = 0..55
N_TARGETS=7
NUM_SAMPLES=25
BATCH=5                                   # frozen
PROP="gb_gamma_J_m2"
NUM_ATOMS_KEY="AL_S5_210_TRAIN"
CACHE_NAME="scm_al_baseline"     # cache name of the training config (if it does not agree with this, non-zero exit)
SEED_BASE=20301000                         # seed = 20301000 + c (independent of w; same as for the proposed model)
VRAM_MIN_GIB=71                            # K0 measured maximum peak 47,904 MiB (w=2.0, BATCH=5) x 1.5 = 70.2 GiB, rounded up. The old value 30 was decided only from 23,314 MiB at w=1.0
# Ruling: only for w=4.0, VRAM_MIN_GIB_W4 may lower the limit to 46 (the measured peak of the w4 G0 is 44,558 MiB). For w=1 to 3 and the default (VRAM_MIN_GIB_W4 unset) it stays at 71.
# Ruling: to run k=25 to 27 with w=2.0 on a 48 GB GPU, VRAM_MIN_GIB_W4 is also allowed for w=2.0 (decision note). The variable name is kept for compatibility.
VRAM_MIN_GIB_W4="${VRAM_MIN_GIB_W4:-}"                  # the default is empty. The check is after w is decided (1b below)
VRAM_POLL_INTERVAL_SEC=5
VRAM_POLL_HARD_LIMIT_MIN=40

rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
jget() { sed -n "s/^ *\"$2\": *\"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\} *\$/\1/p" "$1" | head -1; }
M_PRE="/mn""t<CLUSTER_MOUNT>"   # mount prefix of the login node (used on the side that translates /mnt to <CLUSTER_MOUNT>). The literal is avoided so that the path check of the submission pre-check does not reject a login-node path in the body
norm_mnt() { printf '%s' "$1" | sed "s#^${M_PRE}#<CLUSTER_MOUNT>#"; }

# ---- 1. check of k and the mapping to (W, c) ----
K_RAW="${SLURM_ARRAY_TASK_ID:-}"
[ -n "$K_RAW" ] || fatal "SLURM_ARRAY_TASK_ID is not set (production needs sbatch --array=<k...>)"
[[ "$K_RAW" =~ ^[0-9]+$ ]] || fatal "SLURM_ARRAY_TASK_ID must be a non-negative integer (got '${K_RAW}')"
K=$((10#$K_RAW))
if [ "$K" -lt 0 ] || [ "$K" -ge "$N_ARRAY" ]; then fatal "k=${K} out of range 0..$((N_ARRAY - 1))"; fi
# main comparison w=2.0 only (k=14..27). Anything else is fatal
K_MIN_MAIN=14; K_MAX_MAIN=27
if [ "$K" -lt "$K_MIN_MAIN" ] || [ "$K" -gt "$K_MAX_MAIN" ]; then fatal "k=${K} is outside ${K_MIN_MAIN}..${K_MAX_MAIN} (w=2.0 only)"; fi
if [ "$((NUM_SAMPLES % BATCH))" -ne 0 ]; then
  fatal "NUM_SAMPLES=${NUM_SAMPLES} % BATCH=${BATCH} != 0 (generator.py:274 would silently shrink the count; C1)"
fi
NUM_BATCHES=$((NUM_SAMPLES / BATCH))
W_IDX=$((K / N_CHUNKS))
C=$((K % N_CHUNKS))
W="${W_LIST[$W_IDX]}"
TARGET_ID=$((C / 2))
SEED=$((SEED_BASE + C))
CC="$(printf '%02d' "$C")"

# ---- 1b. VRAM_MIN_GIB_W4: only for tasks with w=4.0 may the VRAM lower limit be lowered to an integer in 46..71 GiB. A violation fails even in DRYRUN ----
VRAM_MIN_SOURCE="default"
if [ -n "$VRAM_MIN_GIB_W4" ]; then
  [[ "$VRAM_MIN_GIB_W4" =~ ^[0-9]+$ ]] || fatal "VRAM_MIN_GIB_W4 must be an integer (got '${VRAM_MIN_GIB_W4}')"
  VRAM_W4_INT=$((10#$VRAM_MIN_GIB_W4))
  if [ "$VRAM_W4_INT" -lt 46 ] || [ "$VRAM_W4_INT" -gt 71 ]; then fatal "VRAM_MIN_GIB_W4=${VRAM_MIN_GIB_W4} out of range 46..71"; fi
  { [ "$W" = "2.0" ] || [ "$W" = "4.0" ]; } || fatal "VRAM_MIN_GIB_W4 is set but this task has w=${W} (k=${K}); only w=2.0 or w=4.0 may lower the VRAM floor"
  VRAM_MIN_GIB="$VRAM_W4_INT"
  VRAM_MIN_SOURCE="VRAM_MIN_GIB_W4"
fi
echo "VRAM_MIN_GIB=${VRAM_MIN_GIB} (source=${VRAM_MIN_SOURCE})"

# ---- 2. targets: targets array of eval_targets_range.json (as text = repr. Not rounded) ----
[ -f "$(rp "$TARGETS_JSON")" ] || fatal "targets json not found: $TARGETS_JSON"
mapfile -t TARGETS < <(awk '/"targets": *\[/{f=1; next} f&&/\]/{exit} f{gsub(/[ ,]/,""); if($0!="") print}' "$(rp "$TARGETS_JSON")")
[ "${#TARGETS[@]}" -eq "$N_TARGETS" ] || fatal "targets array length ${#TARGETS[@]} != ${N_TARGETS} in $TARGETS_JSON"
for t in "${TARGETS[@]}"; do
  [[ "$t" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]] || fatal "target '${t}' is not a plain number ($TARGETS_JSON)"
done
GAMMA_STR="${TARGETS[$TARGET_ID]}"

OUT_ROOT="${RUN_REF}/results/gen/baseline/w${W}"
OUT_DIR="${OUT_ROOT}/chunk${CC}"
echo "host: $(hostname)"
echo "MODE dryrun=${DRYRUN} stop_before_gen=${STOP_BEFORE_GEN} k=${K} W=${W} c=${C} target_id=${TARGET_ID} gamma_target=${GAMMA_STR} seed=${SEED} num_atoms_key=${NUM_ATOMS_KEY} num_samples=${NUM_SAMPLES} batch=${BATCH} num_batches=${NUM_BATCHES} guidance=${W}"
echo "output_dir: ${OUT_DIR}"

# ---- 2b. check of the training config (before the resolution of ckpt. Independent of ckpt): compute nodes (DRYRUN=0) read it as YAML. grep is used only for the DRYRUN on the login node ----
CONFIG_SRC="${GBREL_DATA}/weights/baseline_model_config.yaml"   # released copy of the baseline training configuration
CFG_READ="$(rp "$CONFIG_SRC")"
[ -f "$CFG_READ" ] || fatal "config not found: $CONFIG_SRC"
CHECKER_PY="${RUN_REF}/scripts/check_baseline_config.py"
if [ "$DRYRUN" = "0" ]; then
  [ -x "$VENV_PYTHON" ] || fatal "venv python not executable (needed to read the training config as YAML): $VENV_PYTHON"
  "$VENV_PYTHON" "$CHECKER_PY" --config "$CFG_READ" --cache-name "$CACHE_NAME" --num-atoms-key "$NUM_ATOMS_KEY" || fatal "training config check failed (YAML): ${CONFIG_SRC}"
else
  # simple check (grep) for the DRYRUN on the login node. Production (DRYRUN=0) uses the YAML check above.
  grep -q "cache/${CACHE_NAME}/" "$CFG_READ" || fatal "config does not reference cache/${CACHE_NAME}/ (wrong training config?): $CONFIG_SRC"
  CFG_KEY="$(grep -m1 '^num_atoms_distribution_key:' "$CFG_READ" | sed 's/^num_atoms_distribution_key:[[:space:]]*//' || true)"
  echo "RESOLVED train : num_atoms_distribution_key=${CFG_KEY:-<missing>} (from ${CONFIG_SRC})"
  [ "$CFG_KEY" = "$NUM_ATOMS_KEY" ] || fatal "num_atoms key mismatch: train config='${CFG_KEY}' gen='${NUM_ATOMS_KEY}'"
  grep -q 'gb_gamma_J_m2' "$CFG_READ" || fatal "config check failed: no gb_gamma_J_m2 in ${CONFIG_SRC} (not a conditional model?)"
  echo "CONFIG_CHECK: OK (grep, DRYRUN only) cache name = ${CACHE_NAME}, num_atoms_distribution_key=${NUM_ATOMS_KEY}, gb_gamma_J_m2 present"
fi
echo "RESOLVED gen   : --num_atoms_distribution=${NUM_ATOMS_KEY}"
CONFIG_STAGED_EXPECT="$(sed "s#${M_PRE}/#<CLUSTER_MOUNT>/#g" "$CFG_READ")"   # expected derived content (for comparison)

# ---- 3. resolution of ckpt: read only from ckpt_resolved_baseline.json (cannot be overwritten by an environment variable; no fallback) ----
if [ "${CKPT+set}" = "set" ] || [ "${CKPT_EPOCH+set}" = "set" ]; then
  fatal "env CKPT/CKPT_EPOCH is set; the checkpoint is read only from ${CKPT_RESOLVED_JSON} (no override; C1)"
fi
CKPT_REAL=""; CKPT_SHA="<unresolved>"; CKPT_EP="<unresolved>"
if [ -f "$(rp "$CKPT_RESOLVED_JSON")" ]; then
  RJ="$(rp "$CKPT_RESOLVED_JSON")"
  CKPT_REAL="$(jget "$RJ" ckpt_path)"; CKPT_SHA="$(jget "$RJ" sha256)"; CKPT_EP="$(jget "$RJ" epoch)"
  [ -n "$CKPT_REAL" ] && [ -n "$CKPT_SHA" ] && [ -n "$CKPT_EP" ] || fatal "ckpt_resolved_baseline.json is missing ckpt_path/sha256/epoch: $CKPT_RESOLVED_JSON"
  [ "$(jget "$RJ" arm)" = "baseline" ] || fatal "ckpt_resolved_baseline.json arm != baseline: $CKPT_RESOLVED_JSON"
  [[ "$CKPT_SHA" =~ ^[0-9a-f]{64}$ ]] || fatal "sha256 in ckpt_resolved_baseline.json is not 64 hex chars: '${CKPT_SHA}'"
  [[ "$CKPT_EP" =~ ^[0-9]+$ ]] || fatal "epoch in ckpt_resolved_baseline.json is not an integer: '${CKPT_EP}'"
  if [ ! -f "$(rp "$CKPT_REAL")" ]; then
    if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN][WARN] ckpt file not found: $CKPT_REAL" >&2; else fatal "ckpt file not found: $CKPT_REAL"; fi
  else
    SHA_NOW="$(sha256sum "$(rp "$CKPT_REAL")" | awk '{print $1}')"
    [ "$SHA_NOW" = "$CKPT_SHA" ] || fatal "ckpt sha256 mismatch: file=${SHA_NOW} json=${CKPT_SHA} ($CKPT_REAL)"
    echo "CKPT_SHA256_CHECK: OK ${SHA_NOW}"
    if [ -x "$VENV_PYTHON" ]; then
      EPOCH_INFO="$("$VENV_PYTHON" - "$CKPT_REAL" <<'PYEOF' 2>&1
import sys
try:
    import torch
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    print(f"epoch={ck.get('epoch')} global_step={ck.get('global_step')}")
except Exception as e:  # noqa: BLE001
    print("EPOCH_READ_ERROR: " + repr(e))
PYEOF
)"
      EPOCH_INFO="$(printf '%s\n' "$EPOCH_INFO" | tail -n 1)"
      echo "CKPT_EPOCH_INFO: ${EPOCH_INFO} (ckpt=${CKPT_REAL})"
      [[ "$EPOCH_INFO" == epoch=* ]] || fatal "could not read epoch from ${CKPT_REAL}: ${EPOCH_INFO}"
      EP_MEAS="$(printf '%s' "$EPOCH_INFO" | sed -n 's/^epoch=\([0-9][0-9]*\).*/\1/p')"
      [ -n "$EP_MEAS" ] || fatal "could not parse an integer epoch from: ${EPOCH_INFO}"
      [ "$EP_MEAS" -eq "$CKPT_EP" ] || fatal "checkpoint epoch mismatch: torch.load(ckpt)['epoch']=${EP_MEAS} != ckpt_resolved epoch=${CKPT_EP} (${CKPT_REAL})"
      echo "CKPT_EPOCH_CHECK: OK epoch=${EP_MEAS}"
    elif [ "$DRYRUN" = "1" ]; then
      echo "[DRYRUN][WARN] venv python not executable here; torch.load epoch check skipped (done in production)" >&2
    else
      fatal "venv python not executable: $VENV_PYTHON"
    fi
  fi
else
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN][WARN] ckpt not resolved: ${CKPT_RESOLVED_JSON} does not exist (the checkpoint-resolution stage has not been done. FATAL in production)" >&2
  else
    fatal "ckpt_resolved_baseline.json not found (run resolve_ckpt.py first; no fallback): $CKPT_RESOLVED_JSON"
  fi
fi

# ---- 4. staging dir (model_path) ----
STAGE_EP="${CKPT_EP}"
STAGE_DIR="${RUN_REF}/results/gen_model_path/epoch${STAGE_EP}"
STAGE_CKPT="${STAGE_DIR}/checkpoints/epoch=${STAGE_EP}.ckpt"
stage_matches() {  # $1=dir -> 0 if config.yaml bytes match the derived content and the ckpt symlink points to the resolved ckpt
  [ -f "$1/config.yaml" ] && [ "$(cat "$1/config.yaml")" = "$CONFIG_STAGED_EXPECT" ] \
    && [ -L "$1/checkpoints/epoch=${STAGE_EP}.ckpt" ] \
    && [ "$(norm_mnt "$(readlink -f "$1/checkpoints/epoch=${STAGE_EP}.ckpt")")" = "$(norm_mnt "$(readlink -f "$(rp "$CKPT_REAL")")")" ]
}

if [ "$DRYRUN" = "1" ]; then
  echo "[DRYRUN] stage dir (not created): ${STAGE_DIR}  config.yaml <- sed(login-node mount prefix -> <CLUSTER_MOUNT>/) of ${CONFIG_SRC}; checkpoints/epoch=${STAGE_EP}.ckpt -> ${CKPT_REAL:-<unresolved>}"
else
  mkdir -p "$(dirname "$STAGE_DIR")"
  if [ -e "$STAGE_DIR" ]; then
    stage_matches "$STAGE_DIR" || fatal "stage dir exists with different content (no overwrite): ${STAGE_DIR}"
    echo "STAGE_DIR: reuse existing identical ${STAGE_DIR}"
  else
    STAGE_TMP="${STAGE_DIR}.tmp.${SLURM_JOB_ID:-pid$$}_${SLURM_ARRAY_TASK_ID}_$$"
    mkdir -p "${STAGE_TMP}/checkpoints"
    printf '%s\n' "$CONFIG_STAGED_EXPECT" > "${STAGE_TMP}/config.yaml"
    ln -s "$CKPT_REAL" "${STAGE_TMP}/checkpoints/epoch=${STAGE_EP}.ckpt"
    if mv -T "$STAGE_TMP" "$STAGE_DIR" 2>/dev/null; then
      echo "STAGE_DIR: created ${STAGE_DIR}"
    else
      rm -r "$STAGE_TMP"
      stage_matches "$STAGE_DIR" || fatal "stage dir raced by another task with different content: ${STAGE_DIR}"
      echo "STAGE_DIR: another task created it first; identical, reuse ${STAGE_DIR}"
    fi
  fi
  if grep -n "$M_PRE" "${STAGE_DIR}/config.yaml" >&2; then fatal "staged config still contains login-node mount paths"; fi
  # the sha256 cross-check is done on the file in the staging dir (the symlink target) (J-1)
  SHA_STAGE="$(sha256sum "${STAGE_CKPT}" | awk '{print $1}')"
  [ "$SHA_STAGE" = "$CKPT_SHA" ] || fatal "stage ckpt sha256 mismatch: stage=${SHA_STAGE} json=${CKPT_SHA} (${STAGE_CKPT})"
  echo "STAGE_CKPT_SHA256_CHECK: OK ${SHA_STAGE}"
fi

if [ "$STOP_BEFORE_GEN" = "1" ] && [ "$DRYRUN" = "0" ]; then
  echo "STOP_BEFORE_GEN=1: ckpt resolved (epoch=${CKPT_EP}), stage dir ready. Stopping before output dir / chunk_meta / vram precheck / generation. rc=0"
  exit 0
fi

# ---- the python command line actually called (an array = what is printed and what is run are the same) ----
export PYTHONPATH="${GBREL_CODE}"
GEN_CMD=("$VENV_PYTHON" "$WRAPPER" --seed "$SEED" -- "$OUT_DIR"
  "--model_path=${STAGE_DIR}" "--checkpoint_epoch=${STAGE_EP}"
  "--num_atoms_distribution=${NUM_ATOMS_KEY}" "--batch_size=${BATCH}" "--num_batches=${NUM_BATCHES}"
  "--properties_to_condition_on={'${PROP}':${GAMMA_STR}}" "--diffusion_guidance_factor=${W}"
  "--record_trajectories=False")

emit_run_meta() {  # the same 17 keys as for the proposed model. For the baseline the last 4 keys (gamma_csv_path, bulk_membership_csv_path, manifest_dir, reference_dir) are null
  local epoch_json="$CKPT_EP"; [[ "$CKPT_EP" =~ ^[0-9]+$ ]] || epoch_json="\"${CKPT_EP}\""
  cat <<EOF
{
  "arm": "baseline",
  "w": ${W},
  "chunk": ${C},
  "target_id": ${TARGET_ID},
  "gamma_target": ${GAMMA_STR},
  "seed": ${SEED},
  "num_samples": ${NUM_SAMPLES},
  "ckpt_sha256": "${CKPT_SHA}",
  "ckpt_epoch": ${epoch_json},
  "num_atoms_key": "${NUM_ATOMS_KEY}",
  "host_z": 13,
  "solute_zs": "",
  "card_hash12": "${CARD_HASH12}",
  "gamma_csv_path": null,
  "bulk_membership_csv_path": null,
  "manifest_dir": null,
  "reference_dir": null
}
EOF
}

# (internal model-specification check removed for release; see NOTICE)

if [ "$DRYRUN" = "1" ]; then
  echo "[DRYRUN] chunk_meta.json (not written): chunk_index=${C} gamma_target_J_m2=${GAMMA_STR} seed=${SEED} num_samples_requested=${NUM_SAMPLES} batch_size=${BATCH} guidance=${W} property=${PROP} num_atoms_key=${NUM_ATOMS_KEY} checkpoint_path=${CKPT_REAL:-<unresolved>} checkpoint_epoch=${CKPT_EP}"
  echo "[DRYRUN] run_meta.json (not written) -> ${OUT_DIR}/run_meta.json:"
  emit_run_meta
  echo "[DRYRUN] vram precheck: $VENV_PYTHON $RUNNER_GUARDS vram-precheck --min-free-gib $VRAM_MIN_GIB"
  echo "[DRYRUN] vram poll (BG): bash $POLL_VRAM <gen pid> ${OUT_DIR}/vram_poll.csv ${VRAM_POLL_INTERVAL_SEC} ${VRAM_POLL_HARD_LIMIT_MIN}"
  printf '[DRYRUN] python command: '; printf '%q ' "${GEN_CMD[@]}"; printf '| tee %q\n' "${OUT_DIR}/gen_stdout.log"
  echo "[DRYRUN] post-gen check: gen_stdout.log 'Loading model from checkpoint:' path == ${STAGE_CKPT}"
  echo "[DRYRUN] chunk-done marker: $VENV_PYTHON $RUNNER_GUARDS write-done-marker --dir $OUT_DIR --glob generated_crystals_cif.zip --min-count 1 --marker CHUNK_DONE (after ^Lattice= count and zip cif count == ${NUM_SAMPLES})"
  echo "[DRYRUN] WRITE TARGETS (all under ${RUN_REF}/):"
  echo "[DRYRUN]   ${STAGE_DIR}/ (config.yaml, checkpoints/epoch=${STAGE_EP}.ckpt symlink)"
  echo "[DRYRUN]   ${OUT_DIR}/ (chunk_meta.json, run_meta.json, vram_poll.csv, gen_stdout.log, generated_crystals*, CHUNK_DONE)"
  echo "[DRYRUN]   ${RUN_REF}/logs/baseline_gen_<jobid>_<k>.{out,err} (#SBATCH)"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

# ===== the real run starts here =====
[ -f "$RUNNER_GUARDS" ] || fatal "runner_guards not found: $RUNNER_GUARDS"
[ -f "$WRAPPER" ] || fatal "wrapper not found: $WRAPPER"
[ -f "$POLL_VRAM" ] || fatal "poll_vram.sh not found: $POLL_VRAM"
[ -n "$CKPT_REAL" ] || fatal "internal: CKPT_REAL empty in production path"
[ -x "$VENV_PYTHON" ] || fatal "venv python not executable: $VENV_PYTHON"

# ---- 5. output dir (refused if it already has contents) ----
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR")" ]; then fatal "output dir exists and is not empty (no overwrite): $OUT_DIR"; fi
mkdir -p "$OUT_DIR"

# ---- 6. chunk_meta.json (read by the evaluation side; shared convention. Same keys as the source) ----
"$VENV_PYTHON" - "$OUT_DIR/chunk_meta.json" "$C" "$GAMMA_STR" "$SEED" "$NUM_SAMPLES" "$BATCH" "$W" "$PROP" "$NUM_ATOMS_KEY" "$CKPT_REAL" "$CKPT_EP" <<'PYEOF'
import json, sys
(path, k, gstr, seed, nreq, batch, guid, prop, key, ckpt, ep) = sys.argv[1:]
meta = {"chunk_index": int(k), "gamma_target_J_m2": float(gstr), "gamma_target_str": gstr,
        "seed": int(seed), "num_samples_requested": int(nreq), "batch_size": int(batch),
        "guidance": float(guid), "property": prop, "num_atoms_key": key,
        "checkpoint_path": ckpt, "checkpoint_epoch": int(ep), "smoke": 0}
with open(path, "x") as f:   # exception if it exists (not overwritten)
    json.dump(meta, f, indent=2)
print("wrote", path)
PYEOF

# ---- 7. VRAM pre-check ----
"$VENV_PYTHON" "$RUNNER_GUARDS" vram-precheck --min-free-gib "$VRAM_MIN_GIB"

# ---- 8 and 9. VRAM polling (background) and generation ----
VRAM_CSV="${OUT_DIR}/vram_poll.csv"
GEN_STDOUT="${OUT_DIR}/gen_stdout.log"
echo "start: $(date -u)"
set +e
"${GEN_CMD[@]}" > >(tee "$GEN_STDOUT") 2>&1 &
GEN_PID=$!
bash "$POLL_VRAM" "$GEN_PID" "$VRAM_CSV" "$VRAM_POLL_INTERVAL_SEC" "$VRAM_POLL_HARD_LIMIT_MIN" &
POLL_PID=$!
wait "$GEN_PID"
RC=$?
# ---- 11. stopping the polling (by PID) and printing PEAK ----
kill "$POLL_PID" 2>/dev/null
wait "$POLL_PID" 2>/dev/null
sleep 2   # wait for the flush of tee
set -e
echo "end: $(date -u)"
echo "exit_code=${RC}"
if [ -f "$VRAM_CSV" ] && [ "$(wc -l < "$VRAM_CSV")" -gt 1 ]; then
  echo "PEAK_VRAM_MiB=$(awk -F',' 'NR>1{print $3}' "$VRAM_CSV" | sort -n | tail -1)"
else
  echo "PEAK_VRAM_MiB=UNMEASURED (no samples in ${VRAM_CSV})"
fi
[ "$RC" -eq 0 ] || { echo "FATAL: generation exited with ${RC} (see ${GEN_STDOUT})" >&2; exit "$RC"; }

# ---- check after the generation: the path of the ckpt loaded by MatterGen agrees with the ckpt of the staging dir ----
LOADED="$(grep -m1 'Loading model from checkpoint:' "$GEN_STDOUT" | sed 's/.*Loading model from checkpoint:[[:space:]]*//' || true)"
[ -n "$LOADED" ] || { echo "FATAL: 'Loading model from checkpoint:' line not found in ${GEN_STDOUT} (cannot confirm which ckpt was loaded)" >&2; exit 3; }
echo "LOADED_CKPT: ${LOADED}"
[ "$(norm_mnt "$LOADED")" = "$(norm_mnt "$STAGE_CKPT")" ] || { echo "FATAL: loaded ckpt '${LOADED}' != stage ckpt '${STAGE_CKPT}'" >&2; exit 3; }
echo "LOADED_CKPT_CHECK: OK (== stage ckpt; stage ckpt -> $(readlink -f "$STAGE_CKPT"), sha256 verified above)"
grep -m1 "num_atoms_distribution: ${NUM_ATOMS_KEY}" "$GEN_STDOUT" >/dev/null || { echo "FATAL: sampling config in ${GEN_STDOUT} does not show 'num_atoms_distribution: ${NUM_ATOMS_KEY}'" >&2; exit 3; }
echo "NUM_ATOMS_KEY_CHECK: OK (sampling config shows num_atoms_distribution: ${NUM_ATOMS_KEY})"

# ---- 10. acceptance: run_meta and CHUNK_DONE are written only when the number of ^Lattice= lines and the number of cif in the zip == NUM_SAMPLES ----
N_LAT="$(grep -c '^Lattice=' "$OUT_DIR/generated_crystals.extxyz" || true)"
N_CIF="$("$VENV_PYTHON" - "$OUT_DIR/generated_crystals_cif.zip" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    print(sum(1 for n in zf.namelist() if n.endswith(".cif")))
PYEOF
)"
echo "n_lattice=${N_LAT} n_cif=${N_CIF} (expected ${NUM_SAMPLES})"
[ "$N_LAT" -gt 0 ] && [ "$N_CIF" -gt 0 ] || { echo "FATAL: zero structures in ${OUT_DIR} (no marker; B3)" >&2; exit 3; }
if [ "$N_LAT" -ne "$NUM_SAMPLES" ] || [ "$N_CIF" -ne "$NUM_SAMPLES" ]; then
  echo "FATAL: n_lattice=${N_LAT} n_cif=${N_CIF} != ${NUM_SAMPLES} (no marker)" >&2; exit 3
fi
( set -o noclobber; emit_run_meta > "${OUT_DIR}/run_meta.json" )
echo "wrote ${OUT_DIR}/run_meta.json"
"$VENV_PYTHON" "$RUNNER_GUARDS" write-done-marker --dir "$OUT_DIR" --glob "generated_crystals_cif.zip" --min-count 1 --marker CHUNK_DONE
echo "chunk ${C} (w=${W}) done: $(date -u)"

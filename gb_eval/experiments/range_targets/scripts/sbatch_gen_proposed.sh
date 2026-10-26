#!/bin/bash
# Copy of the range-target version. Source: ${GBREL_CODE}/gb_eval/experiments/gen_eval/scripts/sbatch_gen_proposed.sh.
# Only RUN_DIR (the value of the RUN variable), TARGETS_JSON, CARD_HASH12 and the range of k (14..27) were changed.
# Generation of the proposed model (symmetry constraints, variable length) for Al. 7 targets x 50 structures x w=1.0/2.0/3.0/4.0 (w=4.0 is an extension, k=42..55).
#   1 task = one (w, chunk) = 25 structures. array k=0..55 -> w_idx=k//14, c=k%14, W=[1.0,2.0,3.0,4.0][w_idx].
#
# Description: see the experiment description (acceptance conditions on the checkpoint, the targets and the run records).
# Source: ${GBREL_CODE}/gb_eval/generation_hooks/sbatch_gen_al_proposed.sh (read only; copied and modified)
#
# [This file is not submitted at the data-rebuild stage] Submission is a separate step. Usage:
#   G0 (the 3 jobs of chunk 0): sbatch --array=0,14,28 --time=<..> --exclude=<EXCLUDE_NODES> sbatch_gen_proposed.sh
#   G1 (the remaining 39 jobs):        sbatch --array=1-13,15-27,29-41 ... sbatch_gen_proposed.sh     (--array is intentionally not written in the header)
#   Direct SSH (outside SLURM): ssh <gpu-node> "SLURM_ARRAY_TASK_ID=<k> bash ${GBREL_CODE}/gb_eval/experiments/range_targets/scripts/sbatch_gen_proposed.sh"
#       An unset SLURM_JOB_ID is judged to be a direct SSH, and just before the generation at least 30 GiB of free VRAM is checked with runner_guards.py.
#       Liveness is checked with the pid file / the marker at the end of the log (a plain pgrep -f is not used).
#   DRYRUN=1: nothing is run or created; the resolved values, the check results, the python command line and the write destinations are printed and it exits (continues with a WARN if ckpt_resolved has not been made).
#   SMOKE_GEN=1: smoke test of an intermediate ckpt (chunk 0 only. CKPT=<under the smoke directory of MN>). The output goes to RUN/results/gen_smoke/proposed/ (nothing is written to MN).
#
# Environment variables: SLURM_ARRAY_TASK_ID (required. 0..55) / DRYRUN (0|1) / SMOKE_GEN (0|1) / CONFIG (replacement of the training resolved_config. For validation of the invariant check)
#   The ckpt is read only from ckpt_resolved_proposed.json (it cannot be overwritten by the environment variable CKPT, except for SMOKE_GEN=1).
#
# [Differences from the source] (nothing else is changed)
#   1 PROPOSED_TRAIN -> proposed_train
#   2 ckpt: the fixed epoch=199 is abolished -> read from RUN/results/ckpt_resolved_proposed.json and the sha256 is cross-checked
#   3 targets: the targets array (length 7) of eval_targets_range.json. TARGET_ID=c//2
#   4 N_CHUNKS=14, SEED=20301000+c (independent of w), NUM_SAMPLES=25
#   5 GUIDANCE_SCALE=W   6 NUM_ATOMS_KEY=AL_S5_210_TRAIN   7 output results/gen/proposed/w<W>/chunk<cc>/
#   8 completion marker (CHUNK_DONE when there are 25 cif in the zip)   9 array range 0..55 (extension to w=4.0. The mapping of k=0..41 is unchanged)   10 run_meta.json (17 keys)
#   11 the 3 reference-data paths point to the data of the rebuilt training set. A string agreement with the resolved_config.yaml (train_dataset) of the training directory is checked
#   12 before the generation, the internal model-specification check is called (not included in this release; see NOTICE). It exits non-zero unless it passes.
#      Output is appended to RUN/logs/ (to standard output only in DRYRUN). Not applicable to SMOKE_GEN=1.
# [Author's choices]
#   J-1: the output destination of SMOKE_GEN was changed from under MN (the source) to RUN/results/gen_smoke/proposed/ (nothing is written to MN).
#   J-2: the 3 paths are held as constants and checked for agreement with the resolved_config. The values read from the resolved_config are also written to the print and to run_meta.
#   J-3: --time follows the source (01:43:00) as an initial value. There is no result for this data, so the duration of G0 is measured and overridden on the command line.
#   J-4: read-only inputs are translated to <CLUSTER_MOUNT> so that DRYRUN is possible even on the login node that has no <CLUSTER_MOUNT> (rp function). The write destination is always in the <CLUSTER_MOUNT> notation.
#   J-5: PYTHONDONTWRITEBYTECODE=1 is exported (no __pycache__ is made in MN).
#   J-6: the check is run with the python of the venv of mattergen (/usr/bin/python3 of the compute nodes is 3.6 and cannot run the tool).
# A --time is required on the GPU partition; the shard count = VRAM in GB (shard:40). The execution environment is transcribed verbatim from the source (<CLUSTER_MOUNT>/... from the viewpoint of compute nodes).
#
#SBATCH --job-name=al_proposed_gen
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --time=01:43:00
#SBATCH --gres=shard:40   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/proposed_gen_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/proposed_gen_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
SMOKE_GEN="${SMOKE_GEN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
case "$SMOKE_GEN" in 0|1) ;; *) echo "FATAL: SMOKE_GEN must be 0 or 1 (got '${SMOKE_GEN}')" >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1

# ---- fixed values (<CLUSTER_MOUNT> from the viewpoint of compute nodes. VENV_PYTHON is verbatim from the source) ----
VENV_PYTHON="${GEN_PY}"
GENERATE_SCRIPT="${GBREL_CODE}/scripts/generate_symmetry_constrained.py"
TARGETS_JSON="${GBREL_WORK}/workspace/experiments/range_targets/results/eval_targets_range.json"
RUNNER_GUARDS="${GBREL_CODE}/gb_eval/vendor/bin/runner_guards.py"
RUN_REF="${GBREL_WORK}/workspace/experiments/range_targets"
PROPOSED_TRAIN="${GBREL_WORK}/scm-work/experiments/proposed_train"
CKPT_RESOLVED_JSON="${RUN_REF}/results/ckpt_resolved_proposed.json"

# reference data (rebuilt training set)
GAMMA_CSV="${GBREL_DATA}/training/gamma_dedup.csv"
BULK_MEMBERSHIP_CSV="${GBREL_DATA}/training/bulk_membership.csv"
MANIFEST_DIR="${GBREL_DATA}/training/manifests"

HOST_Z_EXPECT=13
NUM_ATOMS_KEY="AL_S5_210_TRAIN"
CARD_HASH12="471d507f4133"
W_LIST=(1.0 2.0 3.0 4.0)
NUM_SAMPLES=25
N_CHUNKS=14
N_ARRAY=56               # k = 0..55
SEED_BASE=20301000       # seed = 20301000 + c (independent of w)
N_TARGETS=7
VRAM_MIN_GIB="${VRAM_MIN_GIB:-30}"

fatal() { echo "FATAL: $*" >&2; exit 2; }
# translation of read-only inputs on the login node (J-4)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
need_file() {  # $1=path $2=label
  if [ -f "$(rp "$1")" ]; then return 0; fi
  if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN][WARN] $2 not found (FATAL in production): $1" >&2; return 0; fi
  fatal "$2 not found: $1"
}
jget() {  # $1=file $2=key -> value (quotes of the string are removed. For JSON with one key per line with indentation)
  sed -n "s/^ *\"$2\": *\"\{0,1\}\([^\",]*\)\"\{0,1\},\{0,1\} *\$/\1/p" "$1" | head -1
}
yaml_train_ds() {  # $1=yaml $2=key -> data_module.train_dataset.<key>
  awk -v key="$2" '
    /^  train_dataset:/ {f=1; next}
    f && /^  [^ ]/ {f=0}
    f && $1==key":" {print $2; exit}' "$1"
}
M_PRE="/mn""t<CLUSTER_MOUNT>"   # mount prefix of the login node (used on the side that translates /mnt to <CLUSTER_MOUNT>). The literal is avoided so that the path check of the submission pre-check does not reject a login-node path in the body
norm_mnt() { printf '%s' "$1" | sed "s#^${M_PRE}#<CLUSTER_MOUNT>#"; }

# ---- check of the environment variables (not silently overwritten) ----
if [ "${SCM_HOST_Z+set}" = "set" ] && [ "$SCM_HOST_Z" != "$HOST_Z_EXPECT" ]; then
  fatal "SCM_HOST_Z='${SCM_HOST_Z}' in env, but this Al run requires ${HOST_Z_EXPECT} (refusing to override silently)"
fi
export SCM_HOST_Z="$HOST_Z_EXPECT"
if [ "${SCM_SOLUTE_ZS+set}" = "set" ] && [ -n "$SCM_SOLUTE_ZS" ]; then
  fatal "SCM_SOLUTE_ZS='${SCM_SOLUTE_ZS}' in env, but pure Al requires empty"
fi
export SCM_SOLUTE_ZS=
export PYTHONPATH="${GBREL_CODE}"

# ---- check of k and the mapping to (W, c) ----
K_RAW="${SLURM_ARRAY_TASK_ID:-}"
if [ "$SMOKE_GEN" = "1" ]; then K_RAW="${K_RAW:-0}"; fi
[ -n "$K_RAW" ] || fatal "SLURM_ARRAY_TASK_ID is not set (production needs sbatch --array=<k...>, or SLURM_ARRAY_TASK_ID=<k> for a direct node)"
[[ "$K_RAW" =~ ^[0-9]+$ ]] || fatal "SLURM_ARRAY_TASK_ID must be a non-negative integer (got '${K_RAW}')"
K=$((10#$K_RAW))
if [ "$K" -lt 0 ] || [ "$K" -ge "$N_ARRAY" ]; then fatal "k=${K} out of range 0..$((N_ARRAY - 1))"; fi
# main comparison w=2.0 only (k=14..27). Anything else is fatal
K_MIN_MAIN=14; K_MAX_MAIN=27
if [ "$K" -lt "$K_MIN_MAIN" ] || [ "$K" -gt "$K_MAX_MAIN" ]; then fatal "k=${K} is outside ${K_MIN_MAIN}..${K_MAX_MAIN} (w=2.0 only)"; fi
W_IDX=$((K / N_CHUNKS))
C=$((K % N_CHUNKS))
W="${W_LIST[$W_IDX]}"
TARGET_ID=$((C / 2))
SEED=$((SEED_BASE + C))
CC="$(printf '%02d' "$C")"

# ---- targets: targets array of eval_targets_range.json (the text of the JSON is used as it is = repr. Not rounded) ----
need_file "$TARGETS_JSON" "targets json"
[ -f "$(rp "$TARGETS_JSON")" ] || fatal "targets json not found (needed to resolve gamma_target even in DRYRUN): $TARGETS_JSON"
mapfile -t TARGETS < <(awk '/"targets": *\[/{f=1; next} f&&/\]/{exit} f{gsub(/[ ,]/,""); if($0!="") print}' "$(rp "$TARGETS_JSON")")
[ "${#TARGETS[@]}" -eq "$N_TARGETS" ] || fatal "targets array length ${#TARGETS[@]} != ${N_TARGETS} in $TARGETS_JSON"
for t in "${TARGETS[@]}"; do
  [[ "$t" =~ ^-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?$ ]] || fatal "target '${t}' is not a plain number ($TARGETS_JSON)"
done
GAMMA_TARGET="${TARGETS[$TARGET_ID]}"

# ---- resolution of the mode ----
if [ "$SMOKE_GEN" = "1" ]; then
  MODE="smoke"
  [ "$C" -eq 0 ] && [ "$W_IDX" -eq 0 ] || fatal "SMOKE_GEN=1 generates (w=1.0, chunk 0) only (got k=${K})"
  [ -n "${CKPT:-}" ] || fatal "SMOKE_GEN=1 requires CKPT (the smoke checkpoint under ${PROPOSED_TRAIN}/smoke/). No default (C1)."
  case "$CKPT" in
    "${PROPOSED_TRAIN}/smoke/"*) ;;
    *) fatal "SMOKE_GEN=1 requires CKPT under ${PROPOSED_TRAIN}/smoke/ (got '${CKPT}')" ;;
  esac
  CONFIG_SRC="${CONFIG:-${PROPOSED_TRAIN}/smoke/resolved_config.yaml}"
  OUTPUT_ROOT="${RUN_REF}/results/gen_smoke/proposed/w${W}"
  CKPT_SHA="<smoke:unchecked>"; CKPT_EPOCH="<smoke>"
else
  MODE="proposed"
  if [ "${CKPT+set}" = "set" ]; then fatal "env CKPT is set ('${CKPT}'); production reads the checkpoint only from ${CKPT_RESOLVED_JSON} (no override; C1)"; fi
  CONFIG_SRC="${CONFIG:-${PROPOSED_TRAIN}/resolved_config.yaml}"
  OUTPUT_ROOT="${RUN_REF}/results/gen/proposed/w${W}"
  CKPT=""; CKPT_SHA="<unresolved>"; CKPT_EPOCH="<unresolved>"
  if [ -f "$(rp "$CKPT_RESOLVED_JSON")" ]; then
    RJ="$(rp "$CKPT_RESOLVED_JSON")"
    CKPT="$(jget "$RJ" ckpt_path)"; CKPT_SHA="$(jget "$RJ" sha256)"; CKPT_EPOCH="$(jget "$RJ" epoch)"
    [ -n "$CKPT" ] && [ -n "$CKPT_SHA" ] && [ -n "$CKPT_EPOCH" ] || fatal "ckpt_resolved_proposed.json is missing ckpt_path/sha256/epoch: $CKPT_RESOLVED_JSON"
    [ "$(jget "$RJ" arm)" = "proposed" ] || fatal "ckpt_resolved_proposed.json arm != proposed: $CKPT_RESOLVED_JSON"
    case "$CKPT" in */smoke/*) fatal "production mode refuses a smoke checkpoint (got '${CKPT}')" ;; esac
    [[ "$CKPT_SHA" =~ ^[0-9a-f]{64}$ ]] || fatal "sha256 in ckpt_resolved_proposed.json is not 64 hex chars: '${CKPT_SHA}'"
    [[ "$CKPT_EPOCH" =~ ^[0-9]+$ ]] || fatal "epoch in ckpt_resolved_proposed.json is not an integer: '${CKPT_EPOCH}'"
  else
    if [ "$DRYRUN" = "1" ]; then
      echo "[DRYRUN][WARN] ckpt not resolved: ${CKPT_RESOLVED_JSON} does not exist (the checkpoint-resolution stage has not been done. FATAL in production)" >&2
    else
      fatal "ckpt_resolved_proposed.json not found (run resolve_ckpt.py first; no fallback to last.ckpt/epoch=199): $CKPT_RESOLVED_JSON"
    fi
  fi
fi

OUTPUT_DIR="${OUTPUT_ROOT}/chunk${CC}"
JOBTAG="${SLURM_JOB_ID:-pid$$}"; [ "$DRYRUN" = "1" ] && JOBTAG="dryrun"
CONFIG_RUN="${RUN_REF}/tmp/gen_cfg/resolved_config_${MODE}_w${W}_chunk${CC}_${JOBTAG}.yaml"
RUN_META="${OUTPUT_DIR}/run_meta.json"

# ---- existence check (no silent fallback) ----
if [ -n "$CKPT" ]; then need_file "$CKPT" "checkpoint (no fallback to last.ckpt)"; fi
need_file "$CONFIG_SRC" "training resolved_config"
need_file "$GENERATE_SCRIPT" "generate script"
need_file "$GAMMA_CSV" "gamma csv"
need_file "$BULK_MEMBERSHIP_CSV" "bulk membership csv"
if [ ! -d "$(rp "$MANIFEST_DIR")" ]; then
  if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN][WARN] manifest dir not found: $MANIFEST_DIR" >&2; else fatal "manifest dir not found: $MANIFEST_DIR"; fi
fi

# ---- cross-check of the sha256 of the ckpt (non-zero exit unless it agrees with the sha256 of ckpt_resolved) ----
if [ "$MODE" = "proposed" ] && [ -n "$CKPT" ]; then
  if [ -f "$(rp "$CKPT")" ]; then
    SHA_NOW="$(sha256sum "$(rp "$CKPT")" | awk '{print $1}')"
    [ "$SHA_NOW" = "$CKPT_SHA" ] || fatal "ckpt sha256 mismatch: file=${SHA_NOW} json=${CKPT_SHA} ($CKPT)"
    echo "CKPT_SHA256_CHECK: OK ${SHA_NOW}"
  elif [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN][WARN] ckpt file not found, sha256 not checked: $CKPT" >&2
  fi
fi

# ---- cross-check with the values measured on the training side: generation side = the constants above, training side = values read from resolved_config and the training log ----
CFG_READ="$(rp "$CONFIG_SRC")"
TRAIN_STDOUT_LOG="${TRAIN_STDOUT_LOG:-$(dirname "$CONFIG_SRC")/train_stdout.log}"
TRAIN_KEY="<unavailable>"; TRAIN_HOST_Z="<unavailable>"
TRAIN_GAMMA_CSV="<unavailable>"; TRAIN_BULK_CSV="<unavailable>"; TRAIN_MANIFEST="<unavailable>"
if [ -f "$CFG_READ" ]; then
  TRAIN_KEY="$(grep -m1 '^num_atoms_distribution_key:' "$CFG_READ" | sed 's/^num_atoms_distribution_key:[[:space:]]*//' || true)"
  [ -n "$TRAIN_KEY" ] || TRAIN_KEY="<missing-in-config>"
  TRAIN_GAMMA_CSV="$(yaml_train_ds "$CFG_READ" gamma_csv_path)"; [ -n "$TRAIN_GAMMA_CSV" ] || TRAIN_GAMMA_CSV="<missing-in-config>"
  TRAIN_BULK_CSV="$(yaml_train_ds "$CFG_READ" bulk_membership_csv_path)"; [ -n "$TRAIN_BULK_CSV" ] || TRAIN_BULK_CSV="<missing-in-config>"
  TRAIN_MANIFEST="$(yaml_train_ds "$CFG_READ" manifest_dir)"; [ -n "$TRAIN_MANIFEST" ] || TRAIN_MANIFEST="<missing-in-config>"
fi
TLOG_READ="$(rp "$TRAIN_STDOUT_LOG")"
if [ -f "$TLOG_READ" ]; then
  TRAIN_HOST_Z="$(grep -m1 '^\[symmetry_constrained\] SCM_HOST_Z=' "$TLOG_READ" | sed 's/^\[symmetry_constrained\] SCM_HOST_Z=\([0-9]*\).*/\1/' || true)"
  [ -n "$TRAIN_HOST_Z" ] || TRAIN_HOST_Z="<missing-in-log>"
fi
echo "host: $(hostname)"
echo "MODE=${MODE} DRYRUN=${DRYRUN} k=${K} W=${W} c=${C} target_id=${TARGET_ID} gamma_target=${GAMMA_TARGET} seed=${SEED} num_atoms_key=${NUM_ATOMS_KEY} guidance_scale=${W} num_samples=${NUM_SAMPLES}"
echo "RESOLVED gen   : SCM_HOST_Z=${SCM_HOST_Z} SCM_SOLUTE_ZS='${SCM_SOLUTE_ZS}' --host-z ${HOST_Z_EXPECT} num_atoms_distribution_key=${NUM_ATOMS_KEY}"
echo "RESOLVED train : SCM_HOST_Z=${TRAIN_HOST_Z} (from ${TRAIN_STDOUT_LOG})  num_atoms_distribution_key=${TRAIN_KEY} (from ${CONFIG_SRC})"
echo "REFDATA gen    : gamma_csv_path=${GAMMA_CSV}"
echo "REFDATA gen    : bulk_membership_csv_path=${BULK_MEMBERSHIP_CSV}"
echo "REFDATA gen    : manifest_dir=${MANIFEST_DIR}"
echo "REFDATA train  : gamma_csv_path=${TRAIN_GAMMA_CSV}  (resolved_config data_module.train_dataset)"
echo "REFDATA train  : bulk_membership_csv_path=${TRAIN_BULK_CSV}"
echo "REFDATA train  : manifest_dir=${TRAIN_MANIFEST}"
if [ "$TRAIN_KEY" != "<unavailable>" ] && [ "$TRAIN_KEY" != "$NUM_ATOMS_KEY" ]; then
  fatal "num_atoms key mismatch: train='${TRAIN_KEY}' gen='${NUM_ATOMS_KEY}'"
fi
if [ "$TRAIN_HOST_Z" != "<unavailable>" ] && [ "$TRAIN_HOST_Z" != "$SCM_HOST_Z" ]; then
  fatal "SCM_HOST_Z mismatch: train='${TRAIN_HOST_Z}' gen='${SCM_HOST_Z}'"
fi
# invariant: the 3 reference-data paths agree as strings with those at training. Smoke has different training data, so only printed.
if [ "$MODE" = "proposed" ] && [ "$TRAIN_GAMMA_CSV" != "<unavailable>" ]; then
  [ "$TRAIN_GAMMA_CSV" = "$GAMMA_CSV" ] || fatal "gamma_csv_path mismatch: train='${TRAIN_GAMMA_CSV}' gen='${GAMMA_CSV}'"
  [ "$TRAIN_BULK_CSV" = "$BULK_MEMBERSHIP_CSV" ] || fatal "bulk_membership_csv_path mismatch: train='${TRAIN_BULK_CSV}' gen='${BULK_MEMBERSHIP_CSV}'"
  [ "$TRAIN_MANIFEST" = "$MANIFEST_DIR" ] || fatal "manifest_dir mismatch: train='${TRAIN_MANIFEST}' gen='${MANIFEST_DIR}'"
  echo "REFDATA_CHECK: OK (3 paths identical to the training resolved_config; all under data_rebuild/results/)"
fi
if [ "$MODE" = "smoke" ] && [ "$TRAIN_GAMMA_CSV" != "<unavailable>" ] && [ "$TRAIN_GAMMA_CSV" != "$GAMMA_CSV" ]; then
  echo "[smoke][WARN] smoke config's gamma_csv_path differs from the production value (not enforced in smoke)" >&2
fi
if [ "$DRYRUN" != "1" ] && { [ "$TRAIN_KEY" = "<unavailable>" ] || [ "$TRAIN_HOST_Z" = "<unavailable>" ] || [ "$TRAIN_GAMMA_CSV" = "<unavailable>" ]; }; then
  fatal "cannot verify the training-side HOST_Z / num_atoms key / refdata paths (config=${CONFIG_SRC} log=${TRAIN_STDOUT_LOG}); set TRAIN_STDOUT_LOG if the config was copied"
fi
echo "checkpoint: ${CKPT:-<unresolved>}  (epoch=${CKPT_EPOCH} sha256=${CKPT_SHA})"
echo "config(src): $CONFIG_SRC"
echo "config(run): $CONFIG_RUN"
echo "output_dir: $OUTPUT_DIR"

# ---- whether the VRAM free-space check is needed: done on direct-SSH nodes (SLURM_JOB_ID unset). Forced with VRAM_PRECHECK=1 ----
DO_VRAM=0
if [ -z "${SLURM_JOB_ID:-}" ] || [ "${VRAM_PRECHECK:-0}" = "1" ]; then DO_VRAM=1; fi

# ---- the python command line actually called (an array = what is printed and what is run are the same) ----
GEN_CMD=("$VENV_PYTHON" "$GENERATE_SCRIPT"
  --checkpoint "${CKPT:-<unresolved>}"
  --config "$CONFIG_RUN"
  --guidance_scale "$W"
  --seed "$SEED"
  --gamma_target "$GAMMA_TARGET"
  --num_samples "$NUM_SAMPLES"
  --output_path "$OUTPUT_DIR"
  --gamma_csv_path "$GAMMA_CSV"
  --bulk_membership_csv_path "$BULK_MEMBERSHIP_CSV"
  --manifest_dir "$MANIFEST_DIR"
  --num-atoms-distribution-key "$NUM_ATOMS_KEY"
  --host-z "$HOST_Z_EXPECT")

# run_meta.json (17 keys). REFERENCE_DIR is copied after the generation from generation_metrics.json.
emit_run_meta() {  # $1=reference_dir (string. In DRYRUN it is <from generation_metrics.json>)
  local epoch_json="$CKPT_EPOCH"; [[ "$CKPT_EPOCH" =~ ^[0-9]+$ ]] || epoch_json="\"${CKPT_EPOCH}\""
  cat <<EOF
{
  "arm": "proposed",
  "w": ${W},
  "chunk": ${C},
  "target_id": ${TARGET_ID},
  "gamma_target": ${GAMMA_TARGET},
  "seed": ${SEED},
  "num_samples": ${NUM_SAMPLES},
  "ckpt_sha256": "${CKPT_SHA}",
  "ckpt_epoch": ${epoch_json},
  "num_atoms_key": "${NUM_ATOMS_KEY}",
  "host_z": ${HOST_Z_EXPECT},
  "solute_zs": "",
  "card_hash12": "${CARD_HASH12}",
  "gamma_csv_path": "${GAMMA_CSV}",
  "bulk_membership_csv_path": "${BULK_MEMBERSHIP_CSV}",
  "manifest_dir": "${MANIFEST_DIR}",
  "reference_dir": "$1"
}
EOF
}

# (internal model-specification check removed for release; see NOTICE)

if [ "$DRYRUN" = "1" ]; then
  echo "[DRYRUN] env: SCM_HOST_Z=${SCM_HOST_Z} SCM_SOLUTE_ZS='${SCM_SOLUTE_ZS}' PYTHONPATH=${PYTHONPATH} PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"
  if [ -f "$CFG_READ" ]; then
    echo "[DRYRUN] config derivation: would rewrite $(grep -c "$M_PRE" "$CFG_READ" || true) line(s) containing the login-node mount prefix -> <CLUSTER_MOUNT> into ${CONFIG_RUN}"
  else
    echo "[DRYRUN] config derivation: skipped (source config not found)"
  fi
  if [ "$DO_VRAM" = "1" ]; then
    echo "[DRYRUN] vram precheck: $VENV_PYTHON $RUNNER_GUARDS vram-precheck --min-free-gib $VRAM_MIN_GIB"
  else
    echo "[DRYRUN] vram precheck: not run (SLURM job; set VRAM_PRECHECK=1 to force)"
  fi
  printf '[DRYRUN] python command: '; printf '%q ' "${GEN_CMD[@]}"; printf '\n'
  echo "[DRYRUN] post-gen check: generation_metrics.json checkpoint_path == checkpoint above, num_atoms_distribution_key == ${NUM_ATOMS_KEY}, reference_dir contains the rebuilt-data directory"
  echo "[DRYRUN] run_meta.json (not written) -> ${RUN_META}:"
  emit_run_meta "<from generation_metrics.json>"
  echo "[DRYRUN] chunk-done marker: $VENV_PYTHON $RUNNER_GUARDS write-done-marker --dir $OUTPUT_DIR --glob generated_crystals_cif.zip --min-count 1 --marker CHUNK_DONE (after zip cif count == ${NUM_SAMPLES})"
  echo "[DRYRUN] WRITE TARGETS (all under ${RUN_REF}/):"
  echo "[DRYRUN]   ${CONFIG_RUN}"
  echo "[DRYRUN]   ${OUTPUT_DIR}/ (generated_crystals*, generation_metrics.json, run_meta.json, CHUNK_DONE)"
  echo "[DRYRUN]   ${RUN_REF}/logs/proposed_gen_<jobid>_<k>.{out,err} (#SBATCH)"
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

# ===== the real run starts here =====
[ -f "$RUNNER_GUARDS" ] || fatal "runner_guards not found: $RUNNER_GUARDS"
[ -x "$VENV_PYTHON" ] || fatal "venv python not executable: $VENV_PYTHON"
[ -n "$CKPT" ] || fatal "internal: CKPT empty in production path"

if [ "$DO_VRAM" = "1" ]; then
  "$VENV_PYTHON" "$RUNNER_GUARDS" vram-precheck --min-free-gib "$VRAM_MIN_GIB"
fi

# derived copy of the config. Non-zero exit if /mnt remains even in one place. An existing one is not overwritten.
mkdir -p "$(dirname "$CONFIG_RUN")"
if [ -e "$CONFIG_RUN" ]; then fatal "derived config already exists (no overwrite): $CONFIG_RUN"; fi
sed "s#${M_PRE}#<CLUSTER_MOUNT>#g" "$CONFIG_SRC" > "$CONFIG_RUN"
if grep -n '/mnt' "$CONFIG_RUN" >&2; then fatal "derived config still contains /mnt paths: $CONFIG_RUN"; fi
echo "derived config: rewrote $(grep -c "$M_PRE" "$CONFIG_SRC" || true) line(s) -> $CONFIG_RUN"

# The generation itself refuses "non-empty" output directories (no overwriting). An existing non-empty one exits non-zero.
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR")" ]; then fatal "output dir exists and is not empty (no overwrite): $OUTPUT_DIR"; fi
mkdir -p "$OUTPUT_DIR"

echo "MATTERGEN_FILE: $("$VENV_PYTHON" -c 'import mattergen; print(mattergen.__file__)')"
echo "start: $(date -u)"
"${GEN_CMD[@]}"
echo "end: $(date -u)"

# ---- check after the generation: generation_metrics.json written by the generator agrees with the passed values (loaded ckpt, num_atoms key, target, seed) ----
GCM="${OUTPUT_DIR}/generation_metrics.json"
[ -f "$GCM" ] || fatal "generation_metrics.json not written: $GCM"
GCM_CKPT="$(jget "$GCM" checkpoint_path)"
[ "$(norm_mnt "$GCM_CKPT")" = "$(norm_mnt "$CKPT")" ] || fatal "generation_metrics checkpoint_path='${GCM_CKPT}' != ckpt_resolved '${CKPT}'"
[ "$(jget "$GCM" num_atoms_distribution_key)" = "$NUM_ATOMS_KEY" ] || fatal "generation_metrics num_atoms_distribution_key != ${NUM_ATOMS_KEY}"
[ "$(jget "$GCM" seed)" = "$SEED" ] || fatal "generation_metrics seed != ${SEED}"
GCM_REF="$(jget "$GCM" reference_dir)"
[ -n "$GCM_REF" ] || fatal "generation_metrics.json has no reference_dir"
echo "GEN_CHECK: ckpt='${GCM_CKPT}' num_atoms_key=$(jget "$GCM" num_atoms_distribution_key) seed=$(jget "$GCM" seed) reference_dir='${GCM_REF}'"

# ---- completion check: run_meta and the marker are written only when there are NUM_SAMPLES cif in the CIF zip ----
NCIF="$("$VENV_PYTHON" - "$OUTPUT_DIR/generated_crystals_cif.zip" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    print(sum(1 for n in zf.namelist() if n.endswith(".cif")))
PYEOF
)"
echo "n_cif=${NCIF}"
[ "$NCIF" -gt 0 ] || fatal "n_cif=0 in $OUTPUT_DIR (no marker written; B3)"
[ "$NCIF" -eq "$NUM_SAMPLES" ] || fatal "n_cif=${NCIF} != ${NUM_SAMPLES} in $OUTPUT_DIR (no marker written)"
( set -o noclobber; emit_run_meta "$GCM_REF" > "$RUN_META" )
echo "wrote ${RUN_META}"
"$VENV_PYTHON" "$RUNNER_GUARDS" write-done-marker --dir "$OUTPUT_DIR" --glob "generated_crystals_cif.zip" --min-count 1 --marker CHUNK_DONE
echo "chunk ${C} (w=${W}) done: $(date -u)"

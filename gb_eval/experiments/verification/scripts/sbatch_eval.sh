#!/bin/bash
# VRUN copy of the evaluation sbatch script (original: ${GBREL_CODE}/gb_eval/experiments/range_targets/scripts/sbatch_eval.sh; the original is not changed).
# Only these were changed: the layer (LAYER = a regeneration layer of layers.json; the generated inputs GEN_DIR and the outputs EVAL_DIR are under the input_run_root of the layer), the log destination, and the entrance check (stage_gate gen).
#   The calculator, the gate, ckpt_resolved, shard:16 and the steps 0 to 4 are the same as in the original (the original run directory is kept as the reference for the calculator and ckpt_resolved).
#   Usage: LAYER=<layer> sbatch --job-name=al_eval_<model>_<layer> [--dependency=afterok:<generation array job>] --export=ALL,LAYER=<layer> sbatch_eval.sh <proposed|baseline> 2.0
#   DRYRUN=1 LAYER=<layer> bash sbatch_eval.sh <model> 2.0 ... writes nothing; only prints the command line and runs read-only checks (no GPU needed).
# Description: see the experiment description.
# Source (read only; combined into one):
#   ${GBREL_CODE}/gb_eval/generation_hooks/sbatch_eval_al_proposed.sh
#   ${GBREL_CODE}/gb_eval/baseline_generation/sbatch_eval_al_baseline_dedup.sh
#
# [Differences from the source]
#   E-1: arguments ARM and W. Products = results/gen/<model>/w<W>/chunk00..13, output = results/eval/<model>_w<W>/ (gamma_pre.csv, validation.json).
#   E-2: EXPECT_CHUNKS=14, EXPECT_PER_CHUNK=25, EXPECT_ROWS=350.
#   E-3: Step0 checks, common to proposed and baseline, the existence of CHUNK_DONE and run_meta.json of the 14 chunks before ALL_DONE (also the required keys of generation_metrics.json for proposed,
#        and the required keys of chunk_meta.json for baseline).
#   E-4: the calculator for proposed is compute_gamma_pre_proposed.py (a copy that turns non-orthogonality into the error of a row), for baseline compute_gamma_pre_baseline.py (a thin wrapper of the functions of the source).
#        Both take --w and --ckpt-resolved and cross-check with run_meta.
#   E-5: the completion marker is written in eval/<model>_w<W>/ by runner_guards write-done-marker (counts the gamma_pre.csv directly below; the 6 jobs do not write into the same directory).
#   The rest (evaluation environment, GPU connectivity check, the order ALL_DONE -> validation -> computation, shard:16) is the same as the source. --time=01:00:00 is the same initial value as the source (500 structures).
#
# Instruments used (all called read-only):
#   PREP_PY = experiments/gamma_tracking/scripts/prep_and_verify_unrelaxed.py
#   G63V_PY = experiments/earlier_experiment/scripts/check_g63_v.py
#
# a --time is required on the GPU partition; shard count = VRAM in GB (shard:16 for single-point calculations only). The execution environment is transcribed verbatim from the source.
#
#SBATCH --job-name=al_eval   # overridden at submission with --job-name=al_eval_<model>_<layer>
#SBATCH --partition=<GPU_PARTITION>   # edit for your cluster
#SBATCH --time=01:00:00
#SBATCH --gres=shard:16   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_%x_%j.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/eval_%x_%j.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1

ARM="${1:-${ARM:-}}"
W="${2:-${W:-}}"
case "$ARM" in proposed|baseline) ;; *) echo "FATAL: ARM must be proposed or baseline (got '${ARM}')" >&2; exit 2 ;; esac
# ---- layer (layers.json is the only definition. Extra lines needed to switch layers) ----
LAYER="${LAYER:-}"
SC="${GBREL_CODE}/gb_eval/experiments/verification/scripts"
[ -d "$SC" ] || SC="/mn""t${SC}"
source "${SC}/layer_lib.sh"
layer_require "$LAYER"
[ "$ARM" = "$LAYER_ARM" ] || { echo "FATAL: ARM=${ARM} is not the arm of layer ${LAYER} (${LAYER_ARM}; C1)" >&2; exit 2; }
[ "$W" = "$LAYER_W" ] || { echo "FATAL: W=${W} does not match the ws of layer ${LAYER} (${LAYER_W}; a w different from that of the layer is not passed silently)" >&2; exit 2; }

EXPECT_CHUNKS=14
EXPECT_PER_CHUNK=25
EXPECT_ROWS=350
TEMPLATE_A="54.15601152562611"   # common to all Al CNID (template of the model specification)

RUN_DIR="${GBREL_WORK}/workspace/experiments/range_targets"   # Reference for the calculator (compute_gamma_pre_*) and ckpt_resolved (read only). The output goes to the input_run_root of the layer
GEN_DIR="${LAYER_INPUT_ROOT}/results/gen/${ARM}/w${W}"
EVAL_DIR="${LAYER_INPUT_ROOT}/results/eval/${ARM}_w${W}"
OUT_CSV="${EVAL_DIR}/gamma_pre.csv"
G63V_JSON="${EVAL_DIR}/g63v.json"
CKPT_RESOLVED="${RUN_DIR}/results/ckpt_resolved_${ARM}.json"
ARM_LABEL="${ARM}_w${W}"
if [ "$ARM" = "proposed" ]; then
    COMPUTE_PY="${RUN_DIR}/scripts/compute_gamma_pre_proposed.py"
else
    COMPUTE_PY="${RUN_DIR}/scripts/compute_gamma_pre_baseline.py"
fi
PREP_PY="${GBREL_CODE}/gb_eval/experiments/gamma_tracking/scripts/prep_and_verify_unrelaxed.py"
G63V_PY="${GBREL_CODE}/gb_eval/experiments/earlier_experiment/scripts/check_g63_v.py"
RUNNER_GUARDS_DIR="${GBREL_CODE}/gb_eval/vendor/bin"

# the inputs of DRYRUN are read by translating to /mnt so that they can be read even on the login node that has no <CLUSTER_MOUNT> (read only; the write destination is always in the <CLUSTER_MOUNT> notation)
rp() { if [ -e "$1" ]; then printf '%s' "$1"; elif [ -e "/mnt$1" ]; then printf '%s' "/mnt$1"; else printf '%s' "$1"; fi; }
if [ "$DRYRUN" = "1" ]; then
    GEN_READ="$(rp "$GEN_DIR")"; RG_DIR="$(rp "$RUNNER_GUARDS_DIR")"
else
    GEN_READ="$GEN_DIR"; RG_DIR="$RUNNER_GUARDS_DIR"
fi

for f in "${COMPUTE_PY}" "${PREP_PY}" "${G63V_PY}" "${CKPT_RESOLVED}"; do
    if [ ! -f "$(rp "${f}")" ]; then
        if [ "$DRYRUN" = "1" ] && [ "$f" = "$CKPT_RESOLVED" ]; then echo "[DRYRUN][WARN] ckpt_resolved not found (FATAL in production): $f" >&2; continue; fi
        echo "FATAL: required file not found: ${f}" >&2
        exit 2
    fi
done
GEN_PRESENT=1
if [ ! -d "$(rp "${GEN_DIR}")" ]; then
    if [ "$DRYRUN" = "1" ]; then
        # only DRYRUN is downgraded to a warning so that the write destinations can be printed even before the generation (the read-only checks of Step0 and Step1 are skipped). FATAL in production.
        echo "[DRYRUN][WARN] generated dir not found (FATAL in production): ${GEN_DIR}" >&2
        GEN_PRESENT=0
    else
        echo "FATAL: generated dir not found: ${GEN_DIR}" >&2
        exit 2
    fi
fi
for f in "${OUT_CSV}" "${G63V_JSON}"; do
    if [ -e "$(rp "${f}")" ]; then
        echo "FATAL: output already exists (no overwrite): ${f}" >&2
        exit 2
    fi
done

echo "host: $(hostname)"
echo "DRYRUN=${DRYRUN} LAYER=${LAYER} ARM=${ARM} W=${W} GEN_DIR=${GEN_DIR} EVAL_DIR=${EVAL_DIR} OUT_CSV=${OUT_CSV} G63V_JSON=${G63V_JSON} expect chunks=${EXPECT_CHUNKS} per_chunk=${EXPECT_PER_CHUNK} rows=${EXPECT_ROWS}"
echo "start: $(date -u)"

# ---- execution environment (same as the source) ----
if [ "$DRYRUN" != "1" ]; then
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
    conda activate mace-mh
    mkdir -p "${EVAL_DIR}"
    python "${SC}/stage_gate.py" --layer "${LAYER}" --stage gen   # Entrance check: products of the preceding stage (generation), 14 chunks x 25 = 350
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES-<unset>}"
    python - <<'PYCHK'
import sys, torch
ok = torch.cuda.is_available(); n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0) if ok and n else None
print(f"torch.cuda.is_available={ok} device_count={n} device0={name}", flush=True)
if not ok:
    print("FATAL: no GPU available (an MLIP is not run on CPU)", file=sys.stderr); sys.exit(3)
PYCHK
fi

# ============================================================================
# Step0 (E-3): checks read-only the CHUNK_DONE and run_meta.json of all chunks and the required keys that the calculator reads (fails before ALL_DONE is written).
#   The DRYRUN on the login node is run with /usr/bin/python3 (standard library only).
# ============================================================================
echo ""
echo "=== Step0: CHUNK_DONE / run_meta.json / metadata key check (read-only): $(date -u) ==="
if [ "$DRYRUN" = "1" ]; then PYCMD=/usr/bin/python3; else PYCMD=python; fi
if [ "$GEN_PRESENT" = "0" ]; then
    echo "[DRYRUN] Step0 skipped (generated dir absent)"
else
"$PYCMD" - "${GEN_READ}" "${EXPECT_CHUNKS}" "${ARM}" <<'PYEOF'
import json, sys
from pathlib import Path
gen_dir, n, arm = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if arm == "proposed":
    meta_name, required = "generation_metrics.json", ["gamma_target_J_m2", "n_atoms_per_sample", "num_samples_requested", "seed", "checkpoint_path"]
else:
    meta_name, required = "chunk_meta.json", ["gamma_target_J_m2", "seed", "checkpoint_path", "num_samples_requested"]
rm_required = ["arm", "w", "chunk", "target_id", "gamma_target", "seed", "num_samples", "ckpt_sha256", "ckpt_epoch", "num_atoms_key",
               "host_z", "solute_zs", "card_hash12", "gamma_csv_path", "bulk_membership_csv_path", "manifest_dir", "reference_dir"]
bad = []
for i in range(n):
    c = gen_dir / f"chunk{i:02d}"
    if not (c / "CHUNK_DONE").is_file():
        bad.append(f"chunk{i:02d}: CHUNK_DONE missing")
    for name, req in ((meta_name, required), ("run_meta.json", rm_required)):
        p = c / name
        if not p.is_file():
            bad.append(f"chunk{i:02d}: {name} missing"); continue
        keys = set(json.load(open(p)).keys())
        miss = [k for k in req if k not in keys]
        if miss:
            bad.append(f"chunk{i:02d}: {name} missing keys {miss}")
print(f"checked {n} chunks; problems={len(bad)}")
for b in bad[:8]:
    print("  " + b)
if bad:
    print("FATAL: chunk metadata incomplete (ALL_DONE not written)", file=sys.stderr)
    sys.exit(1)
PYEOF
fi

# ============================================================================
# Step1: writing of the ALL_DONE marker (guarded by the existence of the zip files of the 14 chunks). DRYRUN does only assert-artifacts (read only).
# ============================================================================
echo ""
echo "=== Step1: write ALL_DONE marker for ${GEN_DIR}: $(date -u) ==="
if [ "$DRYRUN" = "1" ]; then
    if [ "$GEN_PRESENT" = "1" ]; then
        /usr/bin/python3 "${RG_DIR}/runner_guards.py" assert-artifacts --dir "${GEN_READ}" --glob "chunk*/generated_crystals_cif.zip" --min-count "${EXPECT_CHUNKS}"
    else
        echo "[DRYRUN] assert-artifacts skipped (generated dir absent)"
    fi
    echo "[DRYRUN] Step1 would write ${GEN_DIR}/ALL_DONE"
    echo "[DRYRUN] Step2 (CV) would run: python ${G63V_PY} ${GEN_DIR} --expected-chunks ${EXPECT_CHUNKS} --expected-per-chunk ${EXPECT_PER_CHUNK} --out ${G63V_JSON}"
    if [ "$ARM" = "proposed" ]; then
        echo "[DRYRUN] Step3 would run: python ${PREP_PY} extract --generated-root ${GEN_DIR} --expect-chunks ${EXPECT_CHUNKS} --expect-per-chunk ${EXPECT_PER_CHUNK}   (writes ${GEN_DIR}/chunkNN/cif/)"
        echo "[DRYRUN] Step3 would run: python ${COMPUTE_PY} --output-root ${GEN_DIR} --arm-label ${ARM_LABEL} --w ${W} --ckpt-resolved ${CKPT_RESOLVED} --out ${OUT_CSV} --template-a ${TEMPLATE_A}"
    else
        echo "[DRYRUN] Step3 would run: python ${COMPUTE_PY} --gen-root ${GEN_DIR} --arm-label ${ARM_LABEL} --w ${W} --ckpt-resolved ${CKPT_RESOLVED} --out ${OUT_CSV}"
    fi
    echo "[DRYRUN] Step3 would run: python ${PREP_PY} verify --csv ${OUT_CSV} --expect-rows ${EXPECT_ROWS}"
    echo "[DRYRUN] Step4 would run: python ${RUNNER_GUARDS_DIR}/runner_guards.py write-done-marker --dir ${EVAL_DIR} --glob gamma_pre.csv --min-count 1"
    echo "[DRYRUN] WRITE TARGETS (all under ${LAYER_INPUT_ROOT}/): ${GEN_DIR}/ALL_DONE ${GEN_DIR}/chunkNN/cif/ (proposed only) ${EVAL_DIR}/{gamma_pre.csv,g63v.json,ALL_DONE} ${GBREL_WORK}/workspace/experiments/verification/logs/eval_<jobname>_<jobid>.{out,err}"
    echo "[DRYRUN] done (nothing written)"
    exit 0
fi
python - "${GEN_DIR}" "${EXPECT_CHUNKS}" <<'PYEOF'
import sys
sys.path.insert(0, "${GBREL_CODE}/gb_eval/vendor/bin")
from runner_guards import write_done_marker, GuardError
gen_dir, n = sys.argv[1], int(sys.argv[2])
try:
    marker = write_done_marker(gen_dir, "chunk*/generated_crystals_cif.zip", n, marker="ALL_DONE")
    print(f"wrote {marker}")
except GuardError as e:
    print(f"FATAL: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ============================================================================
# Step2: check C-V (previous-stage check: if it FAILs, do not proceed to the next stage)
# ============================================================================
echo ""
echo "=== Step2: CV (check_g63_v) : $(date -u) ==="
set +e
python "${G63V_PY}" "${GEN_DIR}" --expected-chunks "${EXPECT_CHUNKS}" --expected-per-chunk "${EXPECT_PER_CHUNK}" --out "${G63V_JSON}"
G63V_RC=$?
set -e
cat "${G63V_JSON}"
if [ "${G63V_RC}" -ne 0 ]; then
    echo "FATAL: C-V = FAIL for ${GEN_DIR}. Not proceeding to the gamma calculation. Regeneration or changing the criteria on one's own is prohibited." >&2
    exit 1
fi
echo "CV PASS."

# ============================================================================
# Step3: calculation of gamma before relaxation
# ============================================================================
echo ""
echo "=== Step3: gamma_pre (${ARM}, w=${W}): $(date -u) ==="
if [ "$ARM" = "proposed" ]; then
    python "${PREP_PY}" extract --generated-root "${GEN_DIR}" --expect-chunks "${EXPECT_CHUNKS}" --expect-per-chunk "${EXPECT_PER_CHUNK}"
    python "${COMPUTE_PY}" --output-root "${GEN_DIR}" --arm-label "${ARM_LABEL}" --w "${W}" --ckpt-resolved "${CKPT_RESOLVED}" --out "${OUT_CSV}" --template-a "${TEMPLATE_A}"
else
    python "${COMPUTE_PY}" --gen-root "${GEN_DIR}" --arm-label "${ARM_LABEL}" --w "${W}" --ckpt-resolved "${CKPT_RESOLVED}" --out "${OUT_CSV}"
fi
python "${PREP_PY}" verify --csv "${OUT_CSV}" --expect-rows "${EXPECT_ROWS}"

# ============================================================================
# Step4 (E-5): completion marker (counts the gamma_pre.csv directly below)
# ============================================================================
echo ""
echo "=== Step4: write done marker for ${EVAL_DIR}: $(date -u) ==="
python "${RUNNER_GUARDS_DIR}/runner_guards.py" write-done-marker --dir "${EVAL_DIR}" --glob "gamma_pre.csv" --min-count 1

echo ""
echo "end: $(date -u)"

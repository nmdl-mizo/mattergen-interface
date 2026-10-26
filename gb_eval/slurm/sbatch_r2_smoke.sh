#!/bin/bash
# R2 smoke test: generate 5 structures of one target with the released checkpoint of the proposed model.
# Submit from a directory that has a `logs/` subdirectory:  sbatch gb_eval/slurm/sbatch_r2_smoke.sh   (see docs/reproduce.md for the environment variables)
#SBATCH --job-name=gbrel_r2
#SBATCH --partition=<GPU_PARTITION>            # edit for your cluster
#SBATCH --gres=shard:40              # edit for your cluster
#SBATCH --exclude=<EXCLUDE_NODES>    # edit for your cluster (nodes where a shard job cannot see the GPU)
#SBATCH --time=02:00:00              # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gbrel_r2_%j.out
#SBATCH --error=logs/gbrel_r2_%j.err
set -euo pipefail
: "${GBREL_DATA:?set GBREL_DATA}"
: "${GBREL_CODE:?set GBREL_CODE}"
: "${GEN_PY:?set GEN_PY (python of the generation environment)}"
: "${R2_OUT:?set R2_OUT (output directory; must not exist or be empty)}"
CKPT="$GBREL_DATA/weights/proposed_epoch159.ckpt"
EXPECT=0e236655bd2454e701c2bde95db9a3f80f3aed85e217f9bdf3f9b663201c0e87
GOT=$(sha256sum "$CKPT" | cut -d' ' -f1)
echo "ckpt sha256: $GOT"
if [ "$GOT" != "$EXPECT" ]; then echo "checkpoint sha256 mismatch (expected $EXPECT)" >&2; exit 1; fi
export PYTHONPATH="$GBREL_CODE"
export PYTHONDONTWRITEBYTECODE=1
MG=$("$GEN_PY" -c "import mattergen,sys;print(mattergen.__file__)")
echo "mattergen.__file__: $MG"
case "$MG" in "$GBREL_CODE"/*) ;; *) echo "mattergen is not imported from \$GBREL_CODE" >&2; exit 1 ;; esac
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# Al host (Z=13), no solute: the generator refuses to run otherwise
export SCM_HOST_Z=13
export SCM_SOLUTE_ZS=
mkdir -p "$R2_OUT"
# The `dir` column of the training CSV holds ${GBREL_DATA}; expand it for the generator (envsubst, gettext)
envsubst '${GBREL_DATA}' < "$GBREL_DATA/training/gamma_dedup.csv" > "$R2_OUT/gamma_dedup.expanded.csv"
# target_id=1 (gamma_target = targets[1] of the 7-level range design), seed = 20301000 + 2 (chunk 2), 5 structures, guidance scale w = 2.0
"$GEN_PY" "$GBREL_CODE/scripts/generate_symmetry_constrained.py" \
  --checkpoint "$CKPT" \
  --config "$GBREL_DATA/weights/proposed_resolved_config.yaml" \
  --guidance_scale 2.0 --seed 20301002 --gamma_target 1.1321329025690032 --num_samples 5 \
  --output_path "$R2_OUT/gen" \
  --gamma_csv_path "$R2_OUT/gamma_dedup.expanded.csv" \
  --bulk_membership_csv_path "$GBREL_DATA/training/bulk_membership.csv" \
  --manifest_dir "$GBREL_DATA/training/manifests" \
  --num-atoms-distribution-key AL_S5_210_TRAIN --host-z 13
ls -l "$R2_OUT/gen"
"$GEN_PY" - "$R2_OUT/gen/generated_crystals_cif.zip" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    n = sum(1 for x in zf.namelist() if x.endswith(".cif"))
print("n_cif =", n)
sys.exit(0 if n == 5 else 1)
PYEOF

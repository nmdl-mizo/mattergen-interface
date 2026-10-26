#!/bin/bash
# R1 smoke test: relax 3 generated structures (proposed, w=2.0, target_id=1) with MACE-MH-1 and compare gamma with the published rows.
# Submit from a directory that has a `logs/` subdirectory:  sbatch gb_eval/slurm/sbatch_r1_smoke.sh   (see docs/reproduce.md for the environment variables)
#SBATCH --job-name=gbrel_r1
#SBATCH --partition=<GPU_PARTITION>            # edit for your cluster
#SBATCH --gres=shard:40              # edit for your cluster
#SBATCH --exclude=<EXCLUDE_NODES>    # edit for your cluster (nodes where a shard job cannot see the GPU)
#SBATCH --time=01:00:00              # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gbrel_r1_%j.out
#SBATCH --error=logs/gbrel_r1_%j.err
set -euo pipefail
: "${GBREL_DATA:?set GBREL_DATA}"
: "${GBREL_CODE:?set GBREL_CODE}"
: "${GBREL_WORK:?set GBREL_WORK}"
: "${MACE_MODEL_PATH:?set MACE_MODEL_PATH}"
: "${MACE_PY:?set MACE_PY}"
: "${GEN_PY:?set GEN_PY (read by the shared constants module, not run)}"
: "${R1_OUT:?set R1_OUT (output directory)}"
export PYTHONPATH="$GBREL_CODE/gb_eval"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$MACE_PY" "$GBREL_CODE/gb_eval/make_compat_tree.py"
"$MACE_PY" "$GBREL_CODE/gb_eval/smoke/r1_relax_compare.py" --chunk 2 --indices 0,1,2 --out "$R1_OUT"

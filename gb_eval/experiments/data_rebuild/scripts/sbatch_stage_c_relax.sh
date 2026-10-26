#!/bin/bash
# Data rebuild, stage C: relaxation (an array of 31 templates. 1 task = 1 template = 210 structures). Submit to the GPU partition.
# --time is 1.5 times of 210 structures x 10.8 s (the slower measurement) = about 38 min, i.e. 01:00:00.
# Run the GPU pre-check and the job pre-check before submission. Submit after confirming the completion of stage B (the clusters differ, so a dependency does not work).
# Smoke test: only the first N structures with DATA_RELAX_LIMIT=N. Example: DATA_RELAX_LIMIT=5 sbatch --array=0 sbatch_stage_c_relax.sh
#SBATCH --job-name=stage_c
#SBATCH -p <GPU_PARTITION>   # edit for your cluster
#SBATCH --gres=shard:12   # edit for your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-30%8
#SBATCH --output=logs/stage_c_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/stage_c_%A_%a.err   # edit for your cluster (relative to the submit directory)
set -euo pipefail
PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
cd ${GBREL_CODE}/gb_eval/experiments/data_rebuild/scripts
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} limit=${DATA_RELAX_LIMIT:-none}"
"$PY" stage_c_relax_chunk.py

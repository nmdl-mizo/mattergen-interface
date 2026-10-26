#!/bin/bash
# R0: regenerate all 10 figures of the paper from the released files only (CPU).
# Submit from a directory that has a `logs/` subdirectory:  sbatch gb_eval/slurm/sbatch_r0_figures.sh   (see docs/reproduce.md for the environment variables)
#SBATCH --job-name=gbrel_r0
#SBATCH --partition=xx               # edit for your cluster
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/gbrel_r0_%j.out
#SBATCH --error=logs/gbrel_r0_%j.err
set -uo pipefail
: "${GBREL_DATA:?set GBREL_DATA}"
: "${GBREL_CODE:?set GBREL_CODE}"
: "${GBREL_WORK:?set GBREL_WORK (empty or new directory; the figure scripts refuse to overwrite existing PNG files)}"
: "${MACE_MODEL_PATH:?set MACE_MODEL_PATH (imported by the shared constants module; not used by the figures)}"
: "${MACE_PY:?set MACE_PY (python of the evaluation environment)}"
: "${GEN_PY:?set GEN_PY (python of the generation environment; read by the shared constants module, not run)}"
export PYTHONPATH="$GBREL_CODE/gb_eval:$GBREL_CODE/figures/scripts"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
"$MACE_PY" "$GBREL_CODE/gb_eval/make_compat_tree.py" || exit 1
LOG="$GBREL_WORK/r0_logs"
mkdir -p "$LOG"
: > "$LOG/status.tsv"
for s in fig_F1_schematic fig_F2_validity fig_F3_tracking fig_F4_novelty fig_F5_si_w_sweep \
         fig_F6_si_dft_vs_mlip fig_F7_si_representative_structures fig_F8_si_dx_distribution fig_F9_si_dmax_ecdf fig_F10_si_main_w0_tracking; do
  # run through the symlink view: the F1-F4 scripts locate their inputs and outputs relative to their own path
  ( ulimit -v 8000000; timeout 900 "$MACE_PY" "$GBREL_WORK/workspace/paper/figures/scripts/$s.py" > "$LOG/$s.log" 2>&1 )
  rc=$?
  printf '%s\t%s\n' "$s" "$rc" >> "$LOG/status.tsv"
  echo "$s exit=$rc"
done
echo "PNG files written:"
ls -l "$GBREL_WORK/workspace/paper/figures/png/" 2>/dev/null || true

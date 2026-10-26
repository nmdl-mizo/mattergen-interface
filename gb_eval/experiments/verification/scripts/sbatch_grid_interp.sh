#!/bin/bash
# Distances of the G->T and E->T nearest neighbours and of the adjacent dx pairs (array job of 23 tasks. CPU only, fork Pool within a task). No MLIP is used.
# Prerequisite: grid_interp.py select (sbatch_grid_interp_self_check.sh MODE=select) has already created G_list.csv, E_list.csv and task_map.csv.
# Submission is done by the job runner (the author of this file does not submit):
#   sbatch --array=0-22 [--dependency=afterok:<select job ID>] sbatch_grid_interp.sh   (no --time)
#   Aggregation: sbatch --dependency=afterok:<this array job ID> sbatch_grid_interp_agg.sh
# Task -> (set, split): 0-9 = (G, 0-9 of 10 splits) / 10-21 = (E, 0-11 of 12 splits) / 22 = dxpairs. Cross-checked against task_map.csv (saved by select at the output destination).
# Environment variables: OUT_DIR (default results/grid_interp) / DRYRUN=1 (executes and creates nothing; prints the correspondence table, the command line and the output destination. TASK_ID=<k> for only one task)
# Execution environment: the PY and environment variable lines of sbatch_n.sh (RUN/scripts) copied verbatim (conda mace-mh; proven in an earlier job).
#SBATCH --job-name=al_gi
#SBATCH --partition=ix14-23   # edit for your cluster
#SBATCH --cpus-per-task=20
#SBATCH --array=0-22
#SBATCH --output=logs/grid_interp_%x_%A_%a.out   # edit for your cluster (relative to the submit directory)
#SBATCH --error=logs/grid_interp_%x_%A_%a.err   # edit for your cluster (relative to the submit directory)

set -euo pipefail

DRYRUN="${DRYRUN:-0}"
case "$DRYRUN" in 0|1) ;; *) echo "FATAL: DRYRUN must be 0 or 1 (got '${DRYRUN}')" >&2; exit 2 ;; esac

PY=${MACE_PY}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

VRUN="${GBREL_WORK}/workspace/experiments/verification"
SCRIPT="${VRUN}/scripts/grid_interp.py"
OUT_DIR="${OUT_DIR:-${VRUN}/results/grid_interp}"
PROCS="${SLURM_CPUS_PER_TASK:-20}"
NG=10; NE=12; NTASK=23

# Task -> SET PART NPARTS SUB OUTF DONEF (the same rule as task_map_rows() of grid_interp.py)
resolve() {
  local t=$1
  if (( t < NG )); then
    SET=G; PART=$t; NPARTS=$NG; SUB=nn; OUTF=$(printf 'nn_G_%02d.csv' "$PART"); DONEF=$(printf 'nn_G_%02d.DONE' "$PART")
  elif (( t < NG + NE )); then
    SET=E; PART=$(( t - NG )); NPARTS=$NE; SUB=nn; OUTF=$(printf 'nn_E_%02d.csv' "$PART"); DONEF=$(printf 'nn_E_%02d.DONE' "$PART")
  elif (( t == NG + NE )); then
    SET=T; PART=0; NPARTS=1; SUB=dxpairs; OUTF=dxpairs.csv; DONEF=dxpairs.DONE
  else
    echo "FATAL: task id $t out of range 0..$(( NTASK - 1 ))" >&2; exit 2
  fi
  if [ "$SUB" = "nn" ]; then
    CMD=("$PY" "$SCRIPT" nn --set "$SET" --part "$PART" --nparts "$NPARTS" --procs "$PROCS" --out-dir "$OUT_DIR")
  else
    CMD=("$PY" "$SCRIPT" dxpairs --procs "$PROCS" --out-dir "$OUT_DIR")
  fi
}

if [ "$DRYRUN" = "1" ]; then
  TID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-}}"
  echo "[DRYRUN] host=$(hostname) tasks=${NTASK} (G ${NG} / E ${NE} / dxpairs 1) procs=${PROCS} out_dir=${OUT_DIR}"
  echo "[DRYRUN] task_id,set,part,nparts,subcommand,out_file  (task_map.csv is saved by select at ${OUT_DIR}/task_map.csv)"
  for (( t = 0; t < NTASK; t++ )); do
    if [ -n "$TID" ] && [ "$TID" != "$t" ]; then continue; fi
    resolve "$t"
    printf '[DRYRUN] %d,%s,%d,%d,%s,%s\n' "$t" "$SET" "$PART" "$NPARTS" "$SUB" "$OUTF"
    printf '[DRYRUN]   command: '; printf '%q ' "${CMD[@]}"; printf '\n'
    echo "[DRYRUN]   WRITE TARGETS: ${OUT_DIR}/${OUTF} ${OUT_DIR}/${DONEF}  ${VRUN}/logs/grid_interp_<jobname>_<jobid>_${t}.{out,err}"
  done
  echo "[DRYRUN] done (nothing executed, nothing created)"
  exit 0
fi

TID="${SLURM_ARRAY_TASK_ID:-}"
[[ "$TID" =~ ^[0-9]+$ ]] || { echo "FATAL: SLURM_ARRAY_TASK_ID is not set (sbatch --array=0-22 is required. For DRYRUN use TASK_ID=<k>)" >&2; exit 2; }
resolve "$TID"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-} task=${TID} set=${SET} part=${PART}/${NPARTS} sub=${SUB} procs=${PROCS}"
[ -x "$PY" ] || { echo "FATAL: python not executable: $PY" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "FATAL: script not found: $SCRIPT" >&2; exit 2; }
[ -f "${OUT_DIR}/task_map.csv" ] || { echo "FATAL: ${OUT_DIR}/task_map.csv not found (select is required)" >&2; exit 2; }
[ -f "${OUT_DIR}/G_list.csv" ] && [ -f "${OUT_DIR}/E_list.csv" ] || { echo "FATAL: G_list.csv / E_list.csv not found (select is required)" >&2; exit 2; }
want="${TID},${SET},${PART},${NPARTS},${SUB},${OUTF}"
got="$(awk -F, -v t="$TID" 'NR>1 && $1==t {print $0}' "${OUT_DIR}/task_map.csv" | tr -d '\r')"
[ "$got" = "$want" ] || { echo "FATAL: task_map.csv row '${got}' != '${want}'" >&2; exit 2; }
if [ -f "${OUT_DIR}/${DONEF}" ]; then echo "SKIP: ${DONEF} exists (idempotent)"; exit 0; fi
cd "${VRUN}/scripts"
echo "start: $(date -u)"
"${CMD[@]}"
echo "end: $(date -u)"

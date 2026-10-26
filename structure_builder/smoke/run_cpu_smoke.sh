#!/bin/bash
# structure_builder smoke tests S0, S2, S4, S5, S6 (CPU). Not a SLURM script: run it inside an allocation (or on a node with enough cores).
#   S0  stages 0 -> 1 -> A on the released inputs only; compare the rebuilt k0 structures (31 templates x 10 dx) with the released POSCAR files
#   S2  stage B (all dx); compare the dx_p0.000 removal structures and the frozen indices with the released ones
#   S4  gamma recomputed from the released POSCAR_relaxed + energy.json (stage D1) and compared with the released CSV
#   S5  stage E-1 split rerun, compared with the released split.json
#   S6  stage D2 matching + component selection rerun for one (dx, natoms) group, compared with the released gamma_dedup.csv
# Environment: GBREL_DATA (data bundle), GBREL_CODE (this repository), SB_WORK (EMPTY or new directory), MACE_PY (evaluation-environment python).
# Optional: SB_NPROC (parallel processes, default SLURM_CPUS_PER_TASK or 4), SB_S6_GROUP (default dx_p0.000_n120).
set -uo pipefail
: "${GBREL_DATA:?set GBREL_DATA}"; : "${GBREL_CODE:?set GBREL_CODE}"; : "${SB_WORK:?set SB_WORK}"; : "${MACE_PY:?set MACE_PY}"
SB="$GBREL_CODE/structure_builder"
NPROC="${SB_NPROC:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_CPUS_PER_TASK="$NPROC"
mkdir -p "$SB_WORK/logs" "$SB_WORK/smoke"
LOGDIR="$SB_WORK/logs"
runw() { local name="$1" w="$2"; shift 2; echo "=== $name  [$(date +%H:%M:%S)]"; SB_WORK="$w" "$@" > "$LOGDIR/$name.log" 2>&1; local rc=$?; echo "$name exit=$rc"; printf '%s\t%s\n' "$name" "$rc" >> "$SB_WORK/smoke/status.tsv"; return 0; }
run() { local name="$1"; shift; runw "$name" "$SB_WORK" "$@"; }
: > "$SB_WORK/smoke/status.tsv"
export SB_WORK
run stage0_build "$MACE_PY" "$SB/stage0_templates/cnid_grid_sweep.py" --build-only
NCH=$("$MACE_PY" "$SB/stage1_template_dedup/unified_chunk.py" plan); echo "stage 1: $NCH chunk tasks, $NPROC processes"
run stage1_chunks bash -c "seq 0 $((NCH-1)) | xargs -P $NPROC -I{} '$MACE_PY' '$SB/stage1_template_dedup/unified_chunk.py' chunk {}"
run stage1_aggregate "$MACE_PY" "$SB/stage1_template_dedup/unified_aggregate.py"
run stageA "$MACE_PY" "$SB/stageA_dx/stage_a_templates_dx.py"
run S0_compare "$MACE_PY" "$SB/smoke/s0_compare_k0.py" --released "$GBREL_DATA/training/structures" --out "$SB_WORK/smoke/s0_compare.tsv"
run stageB "$MACE_PY" "$SB/stageB_removal/stage_b_removal.py"
run S2_compare_dx0 "$MACE_PY" "$SB/smoke/s2_compare_removal.py" --released-structures "$GBREL_DATA/training/structures" --released-membership "$GBREL_DATA/training/bulk_membership.csv" --dx dx_p0.000 --out "$SB_WORK/smoke/s2_compare_dx_p0.000.tsv"
run S2_compare_all "$MACE_PY" "$SB/smoke/s2_compare_removal.py" --released-structures "$GBREL_DATA/training/structures" --released-membership "$GBREL_DATA/training/bulk_membership.csv" --dx all --out "$SB_WORK/smoke/s2_compare_all_dx.tsv"
# S4, S5, S6 use their own work directories (independent of the success of S0 - S2)
runw S4_gamma "$SB_WORK/s4work" "$MACE_PY" "$SB/smoke/s4_gamma_recompute.py" --released "$GBREL_DATA/training" --out "$SB_WORK/smoke/s4_compare.tsv"
runw S5_split "$SB_WORK/s5work" "$MACE_PY" "$SB/smoke/s5_split_compare.py" --released "$GBREL_DATA/training" --out "$SB_WORK/smoke/s5_compare.txt"
runw S6_dedup "$SB_WORK/s4work" "$MACE_PY" "$SB/smoke/s6_dedup_group.py" --released "$GBREL_DATA/training" --group "${SB_S6_GROUP:-dx_p0.000_n120}" --out "$SB_WORK/smoke/s6_compare.txt"
echo "=== summary lines"
grep -h '^S[0-9] ' "$SB_WORK"/logs/S*_*.log 2>/dev/null | cut -c1-300
echo "status:"; cat "$SB_WORK/smoke/status.tsv"

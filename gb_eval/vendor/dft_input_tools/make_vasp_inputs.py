#!/usr/bin/env python3
"""Generation of the VASP inputs (20 directories).

Source: see the experiment description.

For each selected[i], makes runs/earlier_internal_run/inputs/b{batch:02d}s{cif_index}/ and writes:
  1. POSCAR: byte copy with shutil.copyfile (never sort or reconstruct)
  2. INCAR: the physical settings (ENCUT, EDIFF, etc.) of INCAR of the source Sigma5_210 calculation, with the
     LSCALAPACK=.FALSE. configuration applied (the configuration that resolved the hang when scaLAPACK is enabled;
     see the earlier VASP verification script sched_step3_main_template.sh:4).
     Earlier revisions were transcribed in the version before this LSCALAPACK addition, and had a defect in which
     the remote smoke test hung with 0 electronic steps
  3. KPOINTS: decided by measurement for each structure (not fixed in advance)
  4. POTCAR: cut out the Al block (lines 1-2228) from the measured path
  5. job script: modelled on the earlier k-convergence script a job script of an earlier k-convergence test
     (elapse=01:30:00; the wrong entry of elapse=01:00:00 was corrected)
"""
import paths as _gbrel  # gb_eval/paths.py (reads GBREL_* environment variables; no defaults)
import filecmp
import json
import math
import shutil
from pathlib import Path

REPO = Path(_gbrel.WORK + '/workspace')
RUN_DIR = REPO / "runs/earlier_internal_run"
SELECTION_JSON = RUN_DIR / "results/selection.json"
INPUTS_DIR = RUN_DIR / "inputs"

POTCAR_SRC = REPO / "experiments/earlier_experiment/mu_crosscheck/results/vasp_reference_denser_k/POTCAR"
POTCAR_AL_END_LINE = 2228  # measured: the first hit of `grep -n 'End of Dataset'`

INCAR_TEXT = """SYSTEM = FixN tilt GB Al static
ISTART = 0
ICHARG = 2
ENCUT = 520
PREC = Accurate
ALGO = Normal
LREAL = Auto
EDIFF = 1E-6
NELM = 200
NSW = 0
IBRION = -1
ISIF = 2
ISMEAR = 1
SIGMA = 0.2
ISPIN = 1
LWAVE = .FALSE.
LCHARG = .FALSE.
NPAR = 4
LSCALAPACK = .FALSE.
"""

KSPACING = 0.25  # A^-1


def parse_lattice_norms(poscar_path: Path):
    """Returns the norms (|a|, |b|, |c|) of the real-size lattice vectors obtained by multiplying the scale factor of the POSCAR."""
    lines = poscar_path.read_text(encoding="ascii").splitlines()
    scale = float(lines[1].strip())
    assert scale > 0, f"negative/zero scale factor not supported: {scale} ({poscar_path})"
    norms = []
    for i in range(3):
        vec = [float(x) for x in lines[2 + i].split()]
        assert len(vec) == 3, f"malformed lattice line: {lines[2 + i]!r}"
        norm = math.sqrt(sum((c * scale) ** 2 for c in vec))
        norms.append(norm)
    return norms


def make_kpoints_content(poscar_path: Path):
    norms = parse_lattice_norms(poscar_path)
    i_normal = max(range(3), key=lambda j: norms[j])
    k = [math.ceil(2.0 * math.pi / norms[j] / KSPACING) for j in range(3)]
    k[i_normal] = 1
    lines = [
        "Gamma-centered mesh (in-plane KSPACING 0.25 A-1, GB-normal axis = 1)",
        "0",
        "Gamma",
        f"{k[0]} {k[1]} {k[2]}",
        "0 0 0",
    ]
    text = "\n".join(lines) + "\n"
    assert all(ord(c) < 128 for c in text), "KPOINTS content is not ASCII-only"
    return text, k, norms, i_normal


def make_potcar(dst: Path):
    with open(POTCAR_SRC, "r", encoding="ascii", errors="strict") as f:
        lines = f.readlines()
    al_block = lines[:POTCAR_AL_END_LINE]
    assert len(al_block) == POTCAR_AL_END_LINE
    with open(dst, "w", encoding="ascii") as f:
        f.writelines(al_block)


def make_sched_script(job_name: str) -> str:
    return f"""#SCHED -N {job_name}
#SCHED -L rscgrp=a-batch
#SCHED -L vnode=1
#SCHED -L vnode-core=16
#SCHED -L elapse=01:30:00
#SCHED -j
#SCHED -S

set -euo pipefail
cd "$SCHED_O_WORKDIR"
module load intel/2023.2
module load mvapich/3.0-intel2023.2
VASP_BIN=<VASP_EXECUTABLE>
which mpirun
test -x "$VASP_BIN" && echo "vasp bin OK: $VASP_BIN"
mpirun -np 16 "$VASP_BIN"
"""


def main():
    sel = json.load(open(SELECTION_JSON))
    selected = sel["selected"]
    assert len(selected) == 20

    kpoints_report = []
    for entry in selected:
        dst_poscar = Path(entry["dst_poscar"])
        target_dir = dst_poscar.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. POSCAR: byte copy
        src_poscar = Path(entry["src_poscar"])
        shutil.copyfile(src_poscar, dst_poscar)
        assert filecmp.cmp(src_poscar, dst_poscar, shallow=False), (
            f"POSCAR byte mismatch: {src_poscar} vs {dst_poscar}"
        )

        # 2. INCAR
        incar_path = target_dir / "INCAR"
        incar_path.write_text(INCAR_TEXT, encoding="ascii")

        # 3. KPOINTS (measured for each structure)
        kpoints_text, k, norms, i_normal = make_kpoints_content(dst_poscar)
        kpoints_path = target_dir / "KPOINTS"
        kpoints_path.write_text(kpoints_text, encoding="ascii")
        kpoints_report.append(
            {
                "dir": target_dir.name,
                "k": k,
                "norms": norms,
                "i_normal": i_normal,
            }
        )

        # 4. POTCAR
        make_potcar(target_dir / "POTCAR")

        # 5. job script
        job_name = f" _{target_dir.name}"
        sched_text = make_sched_script(job_name)
        sched_path = target_dir / "run_vasp.sh"
        sched_path.write_text(sched_text, encoding="ascii")

    print(f"generated {len(selected)} input directories under {INPUTS_DIR}")
    for r in kpoints_report:
        print(r["dir"], "k=", r["k"], "norms=", [round(n, 4) for n in r["norms"]], "i_normal=", r["i_normal"])

    # batch check of the expected KPOINTS
    non_173 = [r for r in kpoints_report if r["k"] != [1, 7, 3]]
    if non_173:
        print("NON-'1 7 3' KPOINTS FOUND (the measured values should be reported and the run stopped):")
        for r in non_173:
            print(" ", r)
    else:
        print("ALL 20 KPOINTS == [1, 7, 3] as expected")


if __name__ == "__main__":
    main()

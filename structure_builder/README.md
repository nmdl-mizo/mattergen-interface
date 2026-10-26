# structure_builder: the training-structure pipeline

This directory contains the scripts that built the 6,510 relaxed training structures (5,509 after de-duplication) used to fine-tune the
models of the paper: from the Sigma5(210)[001] tilt grain boundary of fcc Al, through 31 template structures, ten rigid translations
(dx), symmetric atom-pair removal and MACE-MH-1 relaxation, to the gamma table, the template-level train/val/test split and the template
manifests. The data produced by these scripts in the original work are in the data bundle (`training/`, see `../docs/reproduce.md`);
this directory lets you rebuild them.

Status: the scripts are the original ones with the minimum changes needed to run in this repository layout (paths from environment
variables, no lab-internal tools; every change is listed in `PATCHES.md`). The smoke tests below rebuild parts of the data from the
released inputs and print the differences.
Rebuilding the complete data set (all of stage C, the full D2 matching) was **not** re-run for this release.

## Environment

* `SB_WORK` (required, no default; a `KeyError` is raised if it is unset): a new or empty work directory. Every stage reads and writes below it:
  `stage0/structures/g10/` (stage 0), `stage1/` (stage 1), `results/` and `structures/` (stages A - E), `verification/`.
* `MACE_MODEL_PATH` (required by stage C only): the MACE-MH-1 model file (not distributed here; see `../README.md`).
* python: the evaluation environment of the repository (`../environment-eval.yml`; ASE, pymatgen, scipy; stage C also needs mace-torch and a GPU).
  The stages are run as scripts (`python structure_builder/<stage>/<script>.py`); each script puts `structure_builder/lib` on `sys.path` itself.
* `interface_master` is not installed from PyPI; the copy in `vendor/interfacemaster/` is used (see below).

## Stages

| stage | directory / script | input | output (below `SB_WORK`) | resources |
|---|---|---|---|---|
| 0 | `stage0_templates/cnid_grid_sweep.py --build-only` | `inputs/Al.cif`, `vendor/interfacemaster` | `stage0/structures/g10/p_<i>_<j>/POSCAR` (100 structures, 120 atoms each) | CPU; one process per core (`SLURM_CPUS_PER_TASK`) |
| 1 | `stage1_template_dedup/unified_chunk.py {plan,chunk N}`, `unified_aggregate.py` | stage 0 | `stage1/results/chunk_*.csv`, `stage1/groups_10x10_unified.csv` (31 representatives, `is_rep == 1`) | CPU; 4,950 pairwise matchings (253 tasks of 20 comparisons; 40 - 110 s per task in the original logs) |
| A | `stageA_dx/stage_a_templates_dx.py` | stage 1 | `results/template_map.csv`, `results/template_bulk/*.json`, `structures/<cnid>/<dx_tag>/k0_s<seed>/POSCAR` (31 x 10) | CPU, minutes |
| B | `stageB_removal/stage_b_removal.py` | stage A | `structures/.../k{1..4}_s<seed>/POSCAR`, `results/bulk_membership.csv`, `results/relax_inputs/<cnid>.json` | CPU, minutes |
| C | `stageC_relax/stage_c_relax_chunk.py` (array task = one template, 210 structures) | stage B, `MACE_MODEL_PATH` | `POSCAR_relaxed`, `energy.json` in every structure directory | GPU; about 38 minutes per template in the original run |
| D1 | `stageD_gamma_dedup/stage_d1_build_csv.py` | stage C | `results/gamma_all.csv` | CPU |
| D2 | `stageD_gamma_dedup/stage_d2_match_chunk.py`, `stage_d2_aggregate.py` (`stage_d1_match_trial.py` only for the default pilot-group reuse) | D1 | `results/gamma_dedup.csv` | CPU; 1,000 pairs per array task |
| E | `stageE_split/stage_e_split.py`, `stage_e_manifests.py`, `build_tilt_baseline_csv.py` | D2, stage A/B | `results/split.json`, `results/manifests/<cnid>.json`, CSV files for the baseline | CPU |

`slurm/` has one sbatch example per stage (partition and resource lines are examples: edit them for your cluster).
`lib/` holds the shared modules (`builder_common.py` constants and paths, `config.py`, `removal_builder.py`, `sym_common.py`,
`build_bulk_membership.py`, `dx_lib.py`, and `runner_guards.py`, a small helper used by stage C). `checks/` holds the acceptance checks
of the original work that do not depend on lab-internal files (`stage_a_uniqueness_*.py`, `check_data_rebuild.py`; `check_data_rebuild.py` has not been run in
this layout). `smoke/` holds the verification scripts described below.
`PATCHES.md` is a list of what is not included and what was changed.

## Constraints of the construction

* **Coincidence-site lattice**: Sigma5, rotation axis [001], rotation angle 53.130 deg, interface plane (210)
* **Lattice constant**: a = 4.0366 A (fcc Al conventional cell, MACE-relaxed; inputs/Al.cif has 4.0365507720520695)
* **Template size**: 120 atoms per template
* **CNID grid**: 10 x 10 = 100 rigid translations of the second grain
* **Template de-duplication**: 100 translations -> 31 templates; StructureMatcher stol=0.025, primitive_cell=False, scale=False, attempt_supercell=False (ltol, angle_tol: pymatgen defaults)
* **Cell**: orthogonal; a = 54.156 A (GB normal), b = 4.0366 A, c = 9.026 A (template_map.csv written by stage A)
* **dx (rigid translation along the GB normal)**: grain 2 moved by +dx along Cartesian x, cell a-axis length + 2dx; 10 levels from -0.75 to +0.60 A in steps of 0.15 A
* **Atom removal window**: GB2 core window: fractional x in (0.45, 0.55) (DELTA = 0.05)
* **Removal sets**: k = 1..4 symmetric pairs (2k atoms removed), 5 structures per (template, dx, k), attempts a = 0..19
* **Seeds**: seed = 10000 + 1000*dx_idx + 100*k + a
* **Symmetry operation**: R = diag(+1,-1,-1) (screw-type operation); fit residual upper limit 1e-6 A
* **Relaxation model**: MACE-MH-1, head omat_pbe
* **Relaxation constraint**: 76 atoms fixed (the 38 atoms of each grain closest to fractional x = 0.25 and their images), cell fixed
* **Relaxation optimiser**: FIRE, fmax = 0.02 eV/A, 500 steps
* **Grain-boundary energy**: gamma = (E - n*E_bulk) / (2*|b x c|) * 16.0218 [J/m^2]; E_bulk = -3.7569042674314765 eV/atom
* **Post-relaxation de-duplication**: within the same dx and the same atom count, all pairs are matched (same StructureMatcher setting); unconverged structures excluded; one structure kept per connected component, the one with the smallest (seed, k, cnid); 6,510 -> 5,509
* **Split**: template names in sorted order, default_rng(42).permutation(31), then 25 / 3 / 3 templates (train / val / test)

## interface_master

Stage 0 uses `interfacemaster` (Xie, Shibata, Mizoguchi; MIT licence). The package on PyPI has a single release, `1.1.7`; the working copy used
for the data of this work reports version 1.1.1 and is **not identical** to it: eight of the nine modules are byte-identical,
`interface_generator.py` differs (the PyPI file additionally imports `skopt` and `tqdm.notebook` and has extra arguments for a Bayesian-optimisation
search of the rigid translation, `BO`, `output`, `**kwargs`). The working copy is therefore vendored unchanged in `vendor/interfacemaster/`,
together with the MIT licence text (`vendor/interfacemaster/LICENSE.md`, identical to the `LICENSE.md` of the PyPI wheel; the working copy itself has no licence file).
Please cite: Y. S. Xie, K. Shibata, T. Mizoguchi, arXiv:2211.15173. Stage 0 only needs numpy and pymatgen in addition.

## Smoke tests (rebuild parts of the data and print differences)

`smoke/run_cpu_smoke.sh` runs, on the released inputs only (`GBREL_DATA` = the data bundle): **S0** stages 0 -> 1 -> A and compares the 31 x 10 `k0`
structures with the released ones; **S2** stage B, compared with the released removal structures and `frozen_indices` (`dx_p0.000`, and all dx); **S4** gamma
recomputed from the released `POSCAR_relaxed` + `energy.json` and compared with `gamma_dedup.csv`; **S5** the template split compared with `split.json`;
**S6** the D2 matching and component selection of one (dx, atom count) group (default `dx_p0.000_n120`) compared with the released `gamma_dedup.csv`.
`smoke/s3_relax_twice.py` (GPU) relaxes four released structures twice with stage C and compares the energies with the released `energy.json`.
The scripts print numbers and list the differences; they do not decide pass or fail.

```
export GBREL_DATA=<data bundle> GBREL_CODE=<this repository> MACE_PY=<python of environment-eval.yml> SB_WORK=<new empty directory>
bash structure_builder/smoke/run_cpu_smoke.sh          # inside a CPU allocation with several cores (SB_NPROC processes)
export MACE_MODEL_PATH=<mace-mh-1.model>
$MACE_PY structure_builder/smoke/s3_relax_twice.py --released $GBREL_DATA/training --out s3_compare.tsv   # on a GPU node, with SB_WORK set
```

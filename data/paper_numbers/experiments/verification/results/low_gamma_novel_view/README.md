# Export of low-gamma novel structures for visual inspection (Al main experiment layer=main, w=2.0)

Target: structures that converged after relaxation (relax_rows.status==ok_converged) AND are novel after relaxation (novelty_post_rows.novel==True). Sorted by gamma_relaxed in ascending order per arm. No analysis or interpretation is included.

Generating script: `gb_eval/experiments/verification/scripts/export_low_gamma_novel.py`

## Counts (layer=main, w=2.0, 14 chunks x 25 = 350 requested per arm)

| arm | requested | converged (ok_converged) | converged AND novel_post=True (= population of this table) |
|---|---|---|---|
| proposed | 350 | 350 | 344 |
| baseline | 350 | 330 | 330 |

## proposed top 10 / baseline top 5

| rank | arm | chunk | cif_index | target_id | gamma_target | gamma_relaxed | gamma_unrelaxed | natoms | gb_dx_realized_A | abs(gamma_relaxed-gamma_target) | near (<=0.10) | G | d_rms_min_A | nearest training cnid | nearest dx_A | nearest gamma | equivalent grain boundaries status / op / max_A | file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | proposed | 11 | 2 | 5 | 0.1289 | 0.5350 | 0.5391 | 116 | -0.4948 | 0.4061 | N | N |  |  |  |  | fitted / C2n / 0.000005 | `01_proposed_c11_i02_g0.535.{extxyz,vasp}` |
| 2 | proposed | 11 | 3 | 5 | 0.1289 | 0.5376 | 0.5449 | 112 | -1.4077 | 0.4088 | N | N |  |  |  |  | fitted / C2n / 0.000004 | `02_proposed_c11_i03_g0.538.{extxyz,vasp}` |
| 3 | proposed | 11 | 8 | 5 | 0.1289 | 0.5438 | 0.5479 | 112 | -1.4168 | 0.4150 | N | N |  |  |  |  | fitted / C2n / 0.000003 | `03_proposed_c11_i08_g0.544.{extxyz,vasp}` |
| 4 | proposed | 11 | 22 | 5 | 0.1289 | 0.5442 | 0.5551 | 116 | -0.6472 | 0.4154 | N | N |  |  |  |  | fitted / C2n / 0.000004 | `04_proposed_c11_i22_g0.544.{extxyz,vasp}` |
| 5 | proposed | 10 | 12 | 5 | 0.1289 | 0.5446 | 0.5551 | 112 | -1.4605 | 0.4157 | N | N |  |  |  |  | fitted / C2n / 0.000003 | `05_proposed_c10_i12_g0.545.{extxyz,vasp}` |
| 6 | proposed | 0 | 24 | 0 | 0.7308 | 0.5470 | 0.5494 | 116 | -0.3967 | 0.1838 | N | N |  |  |  |  | fitted / C2n / 0.000005 | `06_proposed_c00_i24_g0.547.{extxyz,vasp}` |
| 7 | proposed | 10 | 3 | 5 | 0.1289 | 0.5717 | 0.5795 | 112 | -1.3276 | 0.4429 | N | N |  |  |  |  | fitted / C2n / 0.000003 | `07_proposed_c10_i03_g0.572.{extxyz,vasp}` |
| 8 | proposed | 10 | 2 | 5 | 0.1289 | 0.5868 | 0.5889 | 116 | -0.5792 | 0.4579 | N | N |  |  |  |  | fitted / C2n / 0.000005 | `08_proposed_c10_i02_g0.587.{extxyz,vasp}` |
| 9 | proposed | 10 | 4 | 5 | 0.1289 | 0.5919 | 0.6212 | 112 | -1.3445 | 0.4631 | N | N |  |  |  |  | fitted / C2n / 0.000003 | `09_proposed_c10_i04_g0.592.{extxyz,vasp}` |
| 10 | proposed | 0 | 4 | 0 | 0.7308 | 0.5946 | 0.6347 | 116 | -0.4847 | 0.1363 | N | N |  |  |  |  | fitted / C2n / 0.000004 | `10_proposed_c00_i04_g0.595.{extxyz,vasp}` |
| 1 | baseline | 0 | 8 | 0 | 0.7308 | 0.0040 | 0.0045 | 120 |  | 0.7268 | N | N |  |  |  |  | no_gb / - / - | `01_baseline_c00_i08_g0.004.{extxyz,vasp}` |
| 2 | baseline | 0 | 12 | 0 | 0.7308 | 0.0060 | 0.0066 | 120 |  | 0.7249 | N | N |  |  |  |  | no_gb / - / - | `02_baseline_c00_i12_g0.006.{extxyz,vasp}` |
| 3 | baseline | 11 | 3 | 5 | 0.1289 | 0.0073 | 0.0085 | 120 |  | 0.1215 | N | N |  |  |  |  | no_gb / - / - | `03_baseline_c11_i03_g0.007.{extxyz,vasp}` |
| 4 | baseline | 0 | 23 | 0 | 0.7308 | 0.0106 | 0.6001 | 114 |  | 0.7202 | N | N |  |  |  |  | no_gb / - / - | `04_baseline_c00_i23_g0.011.{extxyz,vasp}` |
| 5 | baseline | 11 | 13 | 5 | 0.1289 | 0.0129 | 0.5354 | 114 |  | 0.1160 | N | N |  |  |  |  | no_gb / - / - | `05_baseline_c11_i13_g0.013.{extxyz,vasp}` |

## The 3 training structures with the lowest gamma (reference)

| rank | row_index | cnid | k | dx_A | gamma [J/m2] | natoms | file | original file |
|---|---|---|---|---|---|---|---|---|
| 1 | 151 | cnid_0_0 | 0 | 0.45 | 0.5302 | 120 | `train_01_row151_g0.530.vasp` | `<ORIGINAL_ROOT>/workspace/experiments/data_rebuild/structures/cnid_0_0/dx_p0.450/k0_s18000/POSCAR_relaxed` |
| 2 | 132 | cnid_0_0 | 0 | 0.30 | 0.5307 | 120 | `train_02_row132_g0.531.vasp` | `<ORIGINAL_ROOT>/workspace/experiments/data_rebuild/structures/cnid_0_0/dx_p0.300/k0_s17000/POSCAR_relaxed` |
| 3 | 5371 | cnid_7_4 | 2 | -0.45 | 0.5313 | 116 | `train_03_row5371_g0.531.vasp` | `<ORIGINAL_ROOT>/workspace/experiments/data_rebuild/structures/cnid_7_4/dx_m0.450/k2_s12202/POSCAR_relaxed` |

## Definitions of the columns and sources (RUN=experiments/relax_novelty, VER=experiments/verification)

- Population (converged AND novel_post): converged is `status==ok_converged` of `RUN/results/relax/main/<model>_w2.0/chunkNN/relax_rows.csv` (same as aggregate_relax_novelty.py:165-166,178). novel is `novel==True` of `RUN/results/novelty/main/<model>_w2.0/chunkNN/novelty_post_rows.csv` (same: 180). The consistency converged <=> novel non-NA is checked at run time (same as same: 214-215).
- chunk / cif_index: the `chunk` directory name / `cif_index` column of the CSV above. target_id = chunk//2 (checked to agree with the `target_id` column of relax_rows.csv).
- gamma_target, gamma_relaxed, natoms: the columns of the same name of `relax_rows.csv`. gamma_relaxed is also checked to agree with `gamma_relaxed=` in the extxyz header.
- gamma_unrelaxed, gb_dx_realized_A: the columns `gamma_unrelaxed`, `gb_dx_realized_A` of `RUN/results/inputs_manifest_main.csv` (key arm,w=2.0,chunk,cif_index).
- near: |gamma_relaxed-gamma_target| <= 0.10 J/m2 (`RUN/scripts/common.py:79` NEAR_TOL_J. Post-side definition of aggregate_relax_novelty.py:253).
- G: the set of proposed structures that are near AND converged AND novel_post (`VER/scripts/grid_interp.py:219-224`, the actual file `VER/results/grid_interp/G_list.csv`, 90 rows). G exists only for proposed. In the proposed rows of this table, near=Y <=> G=Y is checked at run time. baseline is not a target of G, so the d_rms columns are blank.
- d_rms_min_A, cnid / dx_A / gamma of the nearest training structure: the columns `d_rms_min_A`, `rms_nn_cnid`, `rms_nn_dx_A`, `rms_nn_gamma` of `VER/results/grid_interp/nn_G_00..09.csv` (key chunk,cif_index). These are the values of "the training structure that gives the minimum d_rms over all training structures" (not the `nn_*` columns that give d_min).
- Equivalent grain boundaries (status / op / max_A): the columns `status`, `op_name`, `gb_symfit_max_A` of the stage=post rows of `VER/results/gb_equivalence/gb_symfit_main.csv` (key arm,w=2.0,chunk,cif_index). op_name is one of the 4 classes of the adopted symmetry operation {C2n, m_p, C2t, inv} (gb_symfit_summary.md section 3). C2n of proposed is a structure that satisfies the E decision (theta=0.064 A or less); status=no_gb is a structure for which two grain boundaries could not be identified, and op is blank. Whether the "4 classes" are the same classification as intended in the request is a guess (guess 1).
- Structure files: `<rank 2 digits>_<model>_c<chunk 2 digits>_i<cif_index 2 digits>_g<gamma_relaxed 3 digits>.extxyz` is the frame number `frame_index` (the `frame_index` of relax_rows.csv, checked to agree with `cif_index` of the header) of `RUN/results/relax/main/<model>_w2.0/chunkNN/relaxed.extxyz`, cut out as text. `.vasp` is a POSCAR converted from the same frame with the standard library (Cartesian, sorted by element, with arm, chunk, cif_index and gamma in the comment line). After the conversion the number of atoms and the lattice (9 components, within 1e-9 A) were checked to agree with the header.
- Training reference: the first 3 rows of `RUN/results/train_ref_manifest.csv` sorted by gamma (`gamma_Jm2`) in ascending order. POSCAR is a copy of `poscar_relaxed_path` (<CLUSTER_MOUNT> read as <CLUSTER_MOUNT>). The minimum of gamma over the whole training set is the value of rank 1.

## Result of the conversion check

| file stem | natoms | POSCAR number of atoms and lattice check |
|---|---|---|
| 01_proposed_c11_i02_g0.535 | 116 | OK |
| 02_proposed_c11_i03_g0.538 | 112 | OK |
| 03_proposed_c11_i08_g0.544 | 112 | OK |
| 04_proposed_c11_i22_g0.544 | 116 | OK |
| 05_proposed_c10_i12_g0.545 | 112 | OK |
| 06_proposed_c00_i24_g0.547 | 116 | OK |
| 07_proposed_c10_i03_g0.572 | 112 | OK |
| 08_proposed_c10_i02_g0.587 | 116 | OK |
| 09_proposed_c10_i04_g0.592 | 112 | OK |
| 10_proposed_c00_i04_g0.595 | 116 | OK |
| 01_baseline_c00_i08_g0.004 | 120 | OK |
| 02_baseline_c00_i12_g0.006 | 120 | OK |
| 03_baseline_c11_i03_g0.007 | 120 | OK |
| 04_baseline_c00_i23_g0.011 | 114 | OK |
| 05_baseline_c11_i13_g0.013 | 114 | OK |


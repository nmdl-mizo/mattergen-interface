# Fig. 4, second stage: audit of the implementation of the nearest-train distance

Abbreviations: G.py=gb_eval/experiments/verification/scripts/grid_interp.py, K=gb_eval/experiments/relax_novelty/scripts/common.py, SM=site-packages/pymatgen/analysis/structure_matcher.py (pymatgen 2025.6.14), UNI=gb_eval/experiments/data_rebuild/scripts/stage_d_common.py:29, NN=experiments/verification/results/grid_interp/nn_G_00..09.csv.

## 1. Definition of the distance (basis in the implementation)
The calculator is `StructureMatcher(**UNI, stol=0.6).get_rms_dist(q, t)` (G.py:327, K:161-165). q = a generated structure of G (relaxed), t = a relaxed structure of the training set. UNI = stol 0.025 / primitive_cell False / scale False / attempt_supercell False; ltol 0.2 and angle_tol 5 deg are the defaults (SM:356,358). The nn computation overrides stol with 0.6 to avoid cut-offs (None).

| column | definition | basis |
|---|---|---|
| maxdist_norm_min | minimum over all training structures with the same number of atoms of the "maximum site distance (normalized)" | G.py:295-296, 302 |
| d_min_A | maxdist_norm_min × v、v = (V_q/N)^(1/3) | G.py:123-124, 302 |
| d_rms_at_min_A | RMS x v for the training structure that gives d_min_A (not the minimum of the RMS) | G.py:302 |
| **d_rms_min_A** (main metric) | minimum over all training structures with the same number of atoms of the RMS (normalized) x v | G.py:297, 305-307 |
| nn_row_index / rms_nn_row_index | the training structure of the argmin of the maximum distance / of the RMS argmin, respectively (can be different structures) | G.py:300-301, 305-306 |
| d2_A / d_rms_second_A | second smallest of each series | G.py:304, 308 |
| n_compared / n_censored / all_censored | number of training structures compared with the same number of atoms / number of None returned by get_rms_dist / no valid value | G.py:370-373 |
| n_fit_true | number of True results of UNI.fit(q,t) (stol 0.025). Invariant check; measured 0 | G.py:287, 371, 395-397; summary.md:40 |

- Objective: nn minimizes the maximum site distance, the main metric d_rms_min_A minimizes the RMS. The two can point to different training structures.
- Number of atoms: only training structures with the same number of atoms are compared (G.py:280, 153), not all 4,419 (example: N=114 -> 1,140 structures).
- Assignment: pymatgen LinearAssignment (SM:544-546). The squared-distance matrix is solved with d_2 before removing the mean translation; the removal is after the assignment (SM:548-550). It is not a fixed index correspondence.
- Periodic boundaries: pbc_shortest_vectors (minimum image, SM:544), wrapping of fractional coordinates (SM:488, 496).
- Translation: coarse candidate translations (each site of s1 to a reference site of s2, SM:785-788) -> removal of the mean translation after the assignment (SM:548). No rotation optimization. The lattice correspondence is find_all_mappings (re-taking lattice vectors within ltol/angle_tol, SM:460-468). The list of symmetry operations is not used. No alignment with the 76 bulk atoms (76 appears only in the self-test, G.py:741-744).
- Lattice: Niggli reduction only (SM:980-983), no scaling (SM:699). The distance is the Cartesian distance on the average lattice av_lat of both lattices (SM:478-480), normalized by (V_avg/N)^(1/3) (SM:781). The conversion to Angstrom multiplies by v of the q side (difference of the a axis <= 1.5%, G.py:22-24).
- Compared region: the whole cell, all atoms (not restricted to Master/Slave). The statement that "the 76 bulk atoms are common with the training set, so the difference is decided at the grain boundary" is a claim of the plan; there is no verification with actual data.

## 2. Relation to StructureMatcher (why there is no contradiction)
- The condition for fit() = True is "the **maximum** site distance (normalized) after the assignment and after removing the mean translation <= stol" (SM:622, 626, 792 `max(dist)`). It is not the RMS.
- The rms of get_rms_dist is ||dist||/sqrt(N) (SM:792). For the same mapping, rms <= max holds identically.
- Therefore, a mismatch at stol 0.025 (absolute 0.063-0.066 A) is consistent with a minimum RMS of 0.050 A. A mismatch only means that "there is no training structure whose maximum site distance is <= 0.0655 A"; the RMS can be small because it averages the contribution of a few strongly displaced atoms (near the grain boundary) over N.
- Validation of consistency: n_fit_true = 0 for all 90 structures (summary.md:40). It is checked in the stage-2 aggregation that the minimum of d_min_A (minimum of the maximum distance) exceeds 0.0655 A (Fig4_distance_stats.csv).
- What cannot be compared: stol is dimensionless (the normalization uses the average lattice, and v is taken differently by up to 1.5% in fit and nn); fit is a binary decision with a threshold, nn is a continuous quantity searched up to the threshold 0.6. The x axis of B2 is d_rms_min_A, and the tolerance of stol is not overlaid as a decision line. Only for d_min_A (the maximum distance), as a quantity of the same kind, the tolerance band 0.063-0.066 A can be shown in an SI auxiliary figure (with a note).

## 3. Not confirmed
- The internal implementation of LinearAssignment, pbc_shortest_vectors and find_all_mappings (the body of SM/lattice.py was not read).
- Whether the mapping minimizes the RMS after removing the mean translation exactly (the assignment is decided with d_2 before removal, so it is not necessarily the exact minimum; the consequence is a guess).
- Verification with actual data that d_rms is really decided by differences near the grain boundary (the per-atom contributions are not in the CSV).

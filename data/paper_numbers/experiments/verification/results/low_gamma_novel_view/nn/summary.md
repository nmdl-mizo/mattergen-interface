# Nearest-training-structure matching and mutual matching of the 10 low-gamma novel proposed structures of Al (descriptive; no decision)

- Generated: 2026-10-09T09:13:12Z / host cpu-node / job <job>
- Distance: d_rms = rms_norm x vn13(x) [A] (main), d_max = maxdist_norm x vn13(x) (secondary). x is the query side. matcher stol=0.6 (censored = rms_norm >= 0.6).
- Self matching: rank 1 vs itself d_rms=1.746e-06 A, fit=True

## 1. L->T nearest training structure (nn_L.csv)

| rank | chunk | cif | natoms | gamma_relaxed | n_cmp | n_cens | d_rms_min [A] | train row | cnid | k | dx [A] | gamma_train | d_max@that train [A] | d_rms second [A] | second row/cnid | d_max_min [A] | its row/cnid | n_fit_uni |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 11 | 2 | 116 | 0.5350 | 1003 | 0 | 0.0930 | 5371 | cnid_7_4 | 2 | -0.45 | 0.5313 | 0.1141 | 0.0944 | 4004/cnid_5_0 | 0.1082 | 4004/cnid_5_0 | 0 |
| 2 | 11 | 3 | 112 | 0.5376 | 1191 | 0 | 0.1599 | 2743 | cnid_2_4 | 4 | -0.75 | 0.7358 | 0.3173 | 0.1699 | 1834/cnid_1_3 | 0.2528 | 4164/cnid_5_1 | 0 |
| 3 | 11 | 8 | 112 | 0.5438 | 1191 | 0 | 0.1560 | 1834 | cnid_1_3 | 4 | -0.75 | 0.7536 | 0.3175 | 0.1646 | 1295/cnid_0_9 | 0.2456 | 4164/cnid_5_1 | 0 |
| 4 | 11 | 22 | 116 | 0.5442 | 1003 | 0 | 0.0747 | 4172 | cnid_5_1 | 2 | -0.60 | 0.5561 | 0.1243 | 0.0805 | 1289/cnid_0_9 | 0.1193 | 1863/cnid_1_3 | 0 |
| 5 | 10 | 12 | 112 | 0.5446 | 1191 | 0 | 0.1571 | 1834 | cnid_1_3 | 4 | -0.75 | 0.7536 | 0.3233 | 0.1632 | 1295/cnid_0_9 | 0.2411 | 4164/cnid_5_1 | 0 |
| 6 | 0 | 24 | 116 | 0.5470 | 1003 | 0 | 0.0978 | 3651 | cnid_4_0 | 2 | -0.30 | 0.5678 | 0.1083 | 0.1002 | 5236/cnid_6_4 | 0.1083 | 3651/cnid_4_0 | 0 |
| 7 | 10 | 3 | 112 | 0.5717 | 1191 | 0 | 0.1100 | 4526 | cnid_5_3 | 4 | -0.75 | 0.7567 | 0.1778 | 0.1165 | 3792/cnid_4_1 | 0.1778 | 4526/cnid_5_3 | 0 |
| 8 | 10 | 2 | 116 | 0.5868 | 1003 | 0 | 0.0963 | 3101 | cnid_3_0 | 2 | -0.60 | 0.5912 | 0.1273 | 0.0974 | 3115/cnid_3_0 | 0.1162 | 3417/cnid_3_2 | 0 |
| 9 | 10 | 4 | 112 | 0.5919 | 1191 | 0 | 0.1290 | 1834 | cnid_1_3 | 4 | -0.75 | 0.7536 | 0.2135 | 0.1311 | 1295/cnid_0_9 | 0.1972 | 4164/cnid_5_1 | 0 |
| 10 | 0 | 4 | 116 | 0.5946 | 1003 | 0 | 0.1000 | 5236 | cnid_6_4 | 2 | -0.30 | 0.5639 | 0.1652 | 0.1013 | 3651/cnid_4_0 | 0.1357 | 3620/cnid_4_0 | 0 |

- d_rms_min [A]: range 0.0747 - 0.1599, median 0.1050 (valid 10/10)
- cnid of the nearest training structure (minimum d_rms): 8 kinds, frequency cnid_1_3:3, cnid_7_4:1, cnid_2_4:1, cnid_5_1:1, cnid_4_0:1, cnid_5_3:1, cnid_3_0:1, cnid_6_4:1
- dx [A] of the nearest training structure (minimum d_rms): -0.75:5, -0.6:2, -0.3:2, -0.45:1
- Total of n_fit_uni (number of training structures for which fit with stol 0.025 is True, checked for the 2 nearest): 0

## 2. Mutual matching of L (pairs_L.csv)

- Of all 45 pairs, 20 have the same number of atoms (10 within 112, 10 within 116) and 25 have different numbers of atoms and cannot be matched
- Censored pairs: 0
- Pairs for which fit (stol 0.025) is True: 1 / 20
- Number of connected components built from the pairs with fit True: groups of 112: 4/5, groups of 116: 5/5, total 9 (number of distinct structures by the instrument)
- d_rms of the pairs [A]: range 0.0156 - 0.3187, median 0.1483 (valid 20)
- d_max of the pairs [A] (secondary): range 0.0241 - 0.3502, median 0.1753

## 3. Reference values (shown only for comparison. Source experiments/verification/results/grid_interp/summary.md:3-5)

- Median d_rms: G->T 0.1651, E->T 0.1842, dx neighbor pairs 0.4551 A

No decision sentence is written (descriptive, no decision).

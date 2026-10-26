| quantity | n | n_valid | censored/blank | p10 | p25 | p50 | p75 | p90 | min | max |
|---|---|---|---|---|---|---|---|---|---|---|
| [main] d_rms(G->T) [A] | 90 | 90 | 0 | 0.0802 | 0.1071 | 0.1651 | 0.3145 | 0.4446 | 0.0501 | 0.5730 |
| [main] d_rms(E->T) [A] | 120 | 120 | 0 | 0.0375 | 0.0682 | 0.1842 | 0.2237 | 0.2371 | 0.0180 | 0.3741 |
| [main] d_rms(dx neighbor pairs) [A] | 3331 | 3331 | 0 | 0.0557 | 0.3480 | 0.4551 | 0.5318 | 0.5922 | 0.0248 | 0.7790 |
| [secondary] d_max(G->T) [A] | 90 | 90 | 0 | 0.1210 | 0.1522 | 0.2349 | 0.6689 | 1.5375 | 0.0850 | 2.1232 |
| [secondary] d_max(E->T) [A] | 120 | 120 | 0 | 0.0860 | 0.1522 | 0.2018 | 0.2532 | 0.3184 | 0.0463 | 1.1106 |
| [secondary] d_max(dx neighbor pairs) [A] | 3331 | 3331 | 0 | 0.1705 | 1.9470 | 2.4436 | 2.7324 | 2.9618 | 0.0422 | 3.8564 |

Note: d_rms of G/E is d_rms_min_A of nn.csv (the minimum of the rms over all training structures).

| fraction | d_rms count/denominator [main] | d_rms reference value [A] | d_max count/denominator [secondary] | d_max reference value [A] |
|---|---|---|---|---|
| d_min of G < median of the dx neighbor pairs | 81/90 (all G 90) | 0.4551 | 90/90 (all G 90) | 2.4436 |
| d_min of G > median of d_min of E | 40/90 (all G 90; those with all censored 0 not counted) | 0.1842 | 57/90 (all G 90; those with all censored 0 not counted) | 0.2018 |

| nearest training structure of G | value |
|---|---|
| number of kinds of cnid | 24 |
| top 5 (cnid, count) | cnid_1_3:9; cnid_5_1:9; cnid_7_4:6; cnid_6_2:5; cnid_4_0:5 |
| dx_A of the nearest neighbor is the same as the DX10 level nearest to the generated gb_dx | 32/90 |

| delta = gb_dx - nearest DX10 level | n | outside the range | \|delta\|<=0.02 A | fraction | uniform reference |
|---|---|---|---|---|---|
| proposed main experiment, all 350 | 350 | 207 | 31 | 0.0886 | 0.267 |
| G | 90 | 42 | 11 | 0.1222 | 0.267 |

| target | n_G | n_valid (d_rms) | median of d_rms [A] (main) | median of d_min of the d_max series [A] (secondary) |
|---|---|---|---|---|
| t0 | 34 | 34 | 0.1142 | 0.1581 |
| t1 | 22 | 22 | 0.1355 | 0.2044 |
| t2 | 25 | 25 | 0.3388 | 0.5626 |
| t3 | 9 | 9 | 0.4966 | 1.5207 |
| t4 | 0 | 0 |  |  |
| t5 | 0 | 0 |  |  |
| t6 | 0 | 0 |  |  |

| check | value |
|---|---|
| (a) UNI.fit True between G and train | 0 |
| (b) n_same_166 / n_dmax_below_stol / n_disagree | 166 / 166 / 0 (n_judged_same=166 in the json) |
| vertical line stol x (V/N)^(1/3) [A] | 0.0648 |

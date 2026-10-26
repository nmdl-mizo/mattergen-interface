# gb_symfit counts of G2 and E: layer=main

- Generated: 2026-10-06 22:51:52
- Inputs: csv=experiments/verification/results/gb_equivalence/gb_symfit_main.csv manifest=experiments/relax_novelty/results/inputs_manifest_main.csv (manifest 700 rows / csv 1400 rows)
- G2 = status in {fitted,no_gb,multi_gb,disordered} AND gb0_status == two_gb. E = G2 AND gb_symfit_max_A <= theta. theta = 0.064 A (frozen before the analysis).
- The denominator is fixed to n_req (number of rows of the manifest for each (arm,w)). Exclusions and errors are counted as misses. No conditional rates and no rates summed over w are given. stage=post is the main one, pre is for reference.

## per (arm,w): stage=post

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | 2.0 | 350 | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |
| baseline | 2.0 | 350 | 74 | 0.211 [0.172, 0.257] (74/350) | 0 | 0.000 [0.000, 0.011] (0/350) | 74 | 256 | 0 | 0 | 0 | 0 | 20 | 0 | 350 |

## per (arm,w): stage=pre

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | 2.0 | 350 | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |
| baseline | 2.0 | 350 | 74 | 0.211 [0.172, 0.257] (74/350) | 0 | 0.000 [0.000, 0.011] (0/350) | 74 | 256 | 0 | 0 | 0 | 0 | 20 | 0 | 350 |

## per (arm,w,target): stage=post (counts only)

| arm | w | target_id | gamma_target [J/m²] | n_req | G2 | E |
|---|---|---|---|---|---|---|
| proposed | 2.0 | 0 | 0.7308 | 50 | 50 | 50 |
| proposed | 2.0 | 1 | 1.1321 | 50 | 50 | 50 |
| proposed | 2.0 | 2 | 1.5334 | 50 | 50 | 50 |
| proposed | 2.0 | 3 | 1.9347 | 50 | 50 | 50 |
| proposed | 2.0 | 4 | 2.3361 | 50 | 50 | 50 |
| proposed | 2.0 | 5 | 0.1289 | 50 | 50 | 50 |
| proposed | 2.0 | 6 | 2.9380 | 50 | 50 | 50 |
| baseline | 2.0 | 0 | 0.7308 | 50 | 3 | 0 |
| baseline | 2.0 | 1 | 1.1321 | 50 | 8 | 0 |
| baseline | 2.0 | 2 | 1.5334 | 50 | 11 | 0 |
| baseline | 2.0 | 3 | 1.9347 | 50 | 18 | 0 |
| baseline | 2.0 | 4 | 2.3361 | 50 | 22 | 0 |
| baseline | 2.0 | 5 | 0.1289 | 50 | 1 | 0 |
| baseline | 2.0 | 6 | 2.9380 | 50 | 11 | 0 |

## per (arm,w,target): stage=pre (counts only)

| arm | w | target_id | gamma_target [J/m²] | n_req | G2 | E |
|---|---|---|---|---|---|---|
| proposed | 2.0 | 0 | 0.7308 | 50 | 50 | 50 |
| proposed | 2.0 | 1 | 1.1321 | 50 | 50 | 50 |
| proposed | 2.0 | 2 | 1.5334 | 50 | 50 | 50 |
| proposed | 2.0 | 3 | 1.9347 | 50 | 50 | 50 |
| proposed | 2.0 | 4 | 2.3361 | 50 | 50 | 50 |
| proposed | 2.0 | 5 | 0.1289 | 50 | 50 | 50 |
| proposed | 2.0 | 6 | 2.9380 | 50 | 50 | 50 |
| baseline | 2.0 | 0 | 0.7308 | 50 | 3 | 0 |
| baseline | 2.0 | 1 | 1.1321 | 50 | 8 | 0 |
| baseline | 2.0 | 2 | 1.5334 | 50 | 11 | 0 |
| baseline | 2.0 | 3 | 1.9347 | 50 | 18 | 0 |
| baseline | 2.0 | 4 | 2.3361 | 50 | 22 | 0 |
| baseline | 2.0 | 5 | 0.1289 | 50 | 1 | 0 |
| baseline | 2.0 | 6 | 2.9380 | 50 | 11 | 0 |

## Known-value check (--check-known-main. Source structure_filtered/report.md:20-26,32-38,44-49)

| item | expected | measured | decision |
|---|---|---|---|
| proposed w2.0 post n_req | 350 | 350 | PASS |
| proposed w2.0 post G2 | 350 | 350 | PASS |
| proposed w2.0 post E | 350 | 350 | PASS |
| baseline w2.0 post n_req | 350 | 350 | PASS |
| baseline w2.0 post G2 | 74 | 74 | PASS |
| baseline w2.0 post E | 0 | 0 | PASS |
| baseline w2.0 post G2 per target (target 0 to 6) | [3, 8, 11, 18, 22, 1, 11] | [3, 8, 11, 18, 22, 1, 11] | PASS |
| Wilson 95% of the G2 rate of baseline target0 (3 decimals) | [0.021, 0.162] | [0.021, 0.162] | PASS |


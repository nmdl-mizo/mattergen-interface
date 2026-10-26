# gb_symfit counts of G2 and E: layer=baseline_regen1

- Generated: 2026-10-08 10:52:19
- Inputs: csv=experiments/verification/results/gb_equivalence/gb_symfit_baseline_regen1.csv manifest=experiments/verification/results/inputs_manifest_baseline_regen1.csv (manifest 350 rows / csv 700 rows)
- G2 = status in {fitted,no_gb,multi_gb,disordered} AND gb0_status == two_gb. E = G2 AND gb_symfit_max_A <= theta. theta = 0.064 A (frozen before the analysis).
- The denominator is fixed to n_req (number of rows of the manifest for each (arm,w)). Exclusions and errors are counted as misses. No conditional rates and no rates summed over w are given. stage=post is the main one, pre is for reference.

## per (arm,w): stage=post

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2.0 | 350 | 76 | 0.217 [0.177, 0.263] (76/350) | 3 | 0.009 [0.003, 0.025] (3/350) | 76 | 260 | 0 | 0 | 0 | 0 | 14 | 0 | 350 |

## per (arm,w): stage=pre

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2.0 | 350 | 76 | 0.217 [0.177, 0.263] (76/350) | 2 | 0.006 [0.002, 0.021] (2/350) | 76 | 260 | 0 | 0 | 0 | 0 | 14 | 0 | 350 |

## per (arm,w,target): stage=post (counts only)

| arm | w | target_id | gamma_target [J/m²] | n_req | G2 | E |
|---|---|---|---|---|---|---|
| baseline | 2.0 | 0 | 0.7308 | 50 | 6 | 2 |
| baseline | 2.0 | 1 | 1.1321 | 50 | 4 | 0 |
| baseline | 2.0 | 2 | 1.5334 | 50 | 19 | 0 |
| baseline | 2.0 | 3 | 1.9347 | 50 | 13 | 0 |
| baseline | 2.0 | 4 | 2.3361 | 50 | 16 | 0 |
| baseline | 2.0 | 5 | 0.1289 | 50 | 4 | 1 |
| baseline | 2.0 | 6 | 2.9380 | 50 | 14 | 0 |

## per (arm,w,target): stage=pre (counts only)

| arm | w | target_id | gamma_target [J/m²] | n_req | G2 | E |
|---|---|---|---|---|---|---|
| baseline | 2.0 | 0 | 0.7308 | 50 | 6 | 1 |
| baseline | 2.0 | 1 | 1.1321 | 50 | 4 | 0 |
| baseline | 2.0 | 2 | 1.5334 | 50 | 19 | 0 |
| baseline | 2.0 | 3 | 1.9347 | 50 | 13 | 0 |
| baseline | 2.0 | 4 | 2.3361 | 50 | 16 | 0 |
| baseline | 2.0 | 5 | 0.1289 | 50 | 4 | 1 |
| baseline | 2.0 | 6 | 2.9380 | 50 | 14 | 0 |


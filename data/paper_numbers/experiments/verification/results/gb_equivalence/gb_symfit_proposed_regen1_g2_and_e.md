# gb_symfit counts of G2 and E: layer=proposed_regen1

- Generated: 2026-10-07 23:01:59
- Inputs: csv=experiments/verification/results/gb_equivalence/gb_symfit_proposed_regen1.csv manifest=experiments/verification/results/inputs_manifest_proposed_regen1.csv (manifest 350 rows / csv 700 rows)
- G2 = status in {fitted,no_gb,multi_gb,disordered} AND gb0_status == two_gb. E = G2 AND gb_symfit_max_A <= theta. theta = 0.064 A (frozen before the analysis).
- The denominator is fixed to n_req (number of rows of the manifest for each (arm,w)). Exclusions and errors are counted as misses. No conditional rates and no rates summed over w are given. stage=post is the main one, pre is for reference.

## per (arm,w): stage=post

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | 2.0 | 350 | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |

## per (arm,w): stage=pre

| arm | w | n_req | G2 | G2 rate Wilson 95% | E | E rate Wilson 95% | fitted | no_gb | multi_gb | disordered | axis_mismatch | non_al | excluded_pre | error | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| proposed | 2.0 | 350 | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 1.000 [0.989, 1.000] (350/350) | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 350 |

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


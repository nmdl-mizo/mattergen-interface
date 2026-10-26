# Report of the metrics restricted to valid structures (two grain boundaries, equivalent)

- Generated: 2026-10-06 18:52:04 / layer: main (w=2.0) / command: `gb_eval/experiments/relax_novelty/scripts/aggregate_structure_filtered.py --mode main --out-dir experiments/relax_novelty/results/structure_filtered --procs 20 --selected proposed:2.0,baseline:2.0`
- Main comparison: proposed w2.0 vs baseline w2.0. 5 inner targets (n_req=250) and all 7 targets (n_req=350). The denominator is the requested number. Exclusions, unconverged structures and structures that are not X count as misses.
- **G2** = the status of `gb_symfit_main.csv` is one of fitted, no_gb, multi_gb, disordered, and gb0_status == two_gb (judged to have two grain boundaries at stage 0). Judged separately for stage = post (after relaxation) and pre (before relaxation).
- **E** = G2 and gb_symfit_max_A <= theta. **theta = 0.064 A** (frozen: chosen as 0.064 A after looking at the distribution and not changed in this aggregation). The tolerance within which the two grain boundaries overlap under the 2_1 screw symmetry.
- Metrics: (7a)(7b) = number of converged AND X_post / n_req. (2'-X) = converged AND X_post AND |gamma_relaxed-gamma_target| <= 0.10. (4-X) = converged AND novel AND X_post. (6-post-X) = sum of the numbers of connected components of the induced subgraph whose vertices are restricted to S_post intersect X_post / n_req. (1'-X) = median |gamma_relaxed-gamma_target| of the structures with converged AND X_post only (count shown). The pre side (for reference, not used for claims) is given with the same replacement.

## Notes (preregistered; the wording of the manuscript follows them)

1. **Reason for restricting to G2**: the formula of gamma has the denominator 2A and presupposes that "there are two grain boundaries". The gamma of a structure without two grain boundaries is not a quantity that measures the tracking of the target gamma. (Measured: the unconditional numerator of (2') of baseline is 20/250 for the 5 inner targets, of which 16 do not have two grain boundaries (G2_post=False).)
2. **Asymmetry**: G2 and E of proposed are 100% by construction (measured G2_post 350/350, E_post 350/350), so all changes of the proposed - baseline difference in the G2 and E versions come from the decrease on the baseline side. E of baseline is 0/350 (measured; the minimum of gb_symfit_max_A of the 74 structures judged to have two grain boundaries is 0.206 A for post), so the tracking of baseline is read in the G2 version.
3. **Wording**: do not write "baseline cannot keep the structure" but give counts (main, all 7 targets 74/350 = 21%, 5 inner targets 62/250; both are counts of G2_post, before imposing converged). As a sensitivity check, the number of baseline structures with gb_symfit_max_A < 1 A is 6 for post over all 7 targets (5 for the 5 inner targets; of the 7 targets, 6 are converged) and 1 for pre over all 7 targets (all only in rows with status=fitted). theta stays frozen.
4. **Stratification by target**: the number of baseline structures with two grain boundaries is given per target (table below). Grain boundaries disappear more at lower-gamma targets. A pooled conditional rate (e.g. the near-target rate among structures with two grain boundaries) is not given because its denominator is biased by target.

### Number of structures with two grain boundaries per target (post, converged not imposed; the denominator is 50 each)

| target | gamma_target [J/m2] | proposed G2 | baseline G2 | proposed E | baseline E | baseline G2 (pre) | reasons why baseline is not G2 (post): no_gb / multi_gb / disordered / excluded_pre / other |
|---|---|---|---|---|---|---|---|
| 0 | 0.7308 | 50/50 | 3/50 | 50/50 | 0/50 | 3/50 | 43 / 0 / 0 / 4 / 0 |
| 1 | 1.1321 | 50/50 | 8/50 | 50/50 | 0/50 | 8/50 | 41 / 0 / 0 / 1 / 0 |
| 2 | 1.5334 | 50/50 | 11/50 | 50/50 | 0/50 | 11/50 | 39 / 0 / 0 / 0 / 0 |
| 3 | 1.9347 | 50/50 | 18/50 | 50/50 | 0/50 | 18/50 | 30 / 0 / 0 / 2 / 0 |
| 4 | 2.3361 | 50/50 | 22/50 | 50/50 | 0/50 | 22/50 | 27 / 0 / 0 / 1 / 0 |
| 5 (extrapolation) | 0.1289 | 50/50 | 1/50 | 50/50 | 0/50 | 1/50 | 38 / 0 / 0 / 11 / 0 |
| 6 (extrapolation) | 2.9380 | 50/50 | 11/50 | 50/50 | 0/50 | 11/50 | 38 / 0 / 0 / 1 / 0 |

### Rate of two grain boundaries per target (G2_post / 50. Wilson 95% CI)

| target | gamma_target [J/m2] | proposed G2 rate | baseline G2 rate | baseline E rate |
|---|---|---|---|---|
| 0 | 0.7308 | 1.000 [0.929, 1.000] (50/50) | 0.060 [0.021, 0.162] (3/50) | 0.000 [0.000, 0.071] (0/50) |
| 1 | 1.1321 | 1.000 [0.929, 1.000] (50/50) | 0.160 [0.083, 0.285] (8/50) | 0.000 [0.000, 0.071] (0/50) |
| 2 | 1.5334 | 1.000 [0.929, 1.000] (50/50) | 0.220 [0.128, 0.352] (11/50) | 0.000 [0.000, 0.071] (0/50) |
| 3 | 1.9347 | 1.000 [0.929, 1.000] (50/50) | 0.360 [0.241, 0.499] (18/50) | 0.000 [0.000, 0.071] (0/50) |
| 4 | 2.3361 | 1.000 [0.929, 1.000] (50/50) | 0.440 [0.312, 0.577] (22/50) | 0.000 [0.000, 0.071] (0/50) |
| 5 | 0.1289 | 1.000 [0.929, 1.000] (50/50) | 0.020 [0.004, 0.105] (1/50) | 0.000 [0.000, 0.071] (0/50) |
| 6 | 2.9380 | 1.000 [0.929, 1.000] (50/50) | 0.220 [0.128, 0.352] (11/50) | 0.000 [0.000, 0.071] (0/50) |

### Cross-check and confirmation of reference values (F5, F6)

| item | proposed | baseline |
|---|---|---|
| (7a) numerator = #{G2_post} (main, all 7 targets, converged not imposed) | 350/350 | 74/350 |
| (7a-pre) numerator = #{G2_pre} (main, all 7 targets, converged not imposed) | 350/350 | 74/350 |
| (7b) numerator = #{E_post} (main, all 7 targets, converged not imposed) | 350/350 | 0/350 |
| (7b-pre) numerator = #{E_pre} (main, all 7 targets, converged not imposed) | 350/350 | 0/350 |
| (7a) numerator = #{converged AND G2_post} (value with converged imposed) | 350/350 | 74/350 |
| (7b) numerator = #{converged AND E_post} (value with converged imposed) | 350/350 | 0/350 |

## Main comparison: 5 inner targets (n_req=250). proposed w2.0 vs baseline w2.0

### X = G2

| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |
|---|---|---|---|
| (7a) converged ∧ G2_post | 1.000 [0.985, 1.000] (250/250) | 0.248 [0.199, 0.305] (62/250) | 0.752 [0.693, 0.801] |
| (2'-G2) converged AND G2_post AND near the target (gamma after relaxation) | 0.364 [0.307, 0.425] (91/250) | 0.016 [0.006, 0.040] (4/250) | 0.348 [0.286, 0.410] |
| (4-G2) converged AND novel AND G2_post | 0.980 [0.954, 0.991] (245/250) | 0.248 [0.199, 0.305] (62/250) | 0.732 [0.669, 0.783] |
| (6-post-G2) novel, unique, near the target (connected components of the induced subgraph) | 0.360 [0.303, 0.421] (90/250) | 0.016 [0.006, 0.040] (4/250) | 0.344 [0.282, 0.406] |
| (1'-G2) median abs(gamma_relaxed-gamma_target) (converged AND G2_post only; count shown) | 0.174 (n=250) | 0.399 (n=62) | -0.226 |

### X = E

| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |
|---|---|---|---|
| (7b) converged ∧ E_post | 1.000 [0.985, 1.000] (250/250) | 0.000 [0.000, 0.015] (0/250) | 1.000 [0.979, 1.000] |
| (2'-E) converged AND E_post AND near the target (gamma after relaxation) | 0.364 [0.307, 0.425] (91/250) | 0.000 [0.000, 0.015] (0/250) | 0.364 [0.305, 0.425] |
| (4-E) converged AND novel AND E_post | 0.980 [0.954, 0.991] (245/250) | 0.000 [0.000, 0.015] (0/250) | 0.980 [0.950, 0.991] |
| (6-post-E) novel, unique, near the target (connected components of the induced subgraph) | 0.360 [0.303, 0.421] (90/250) | 0.000 [0.000, 0.015] (0/250) | 0.360 [0.301, 0.421] |
| (1'-E) median abs(gamma_relaxed-gamma_target) (converged AND E_post only; count shown) | 0.174 (n=250) | NA (n=0) | NA |

## Main comparison: all 7 targets (n_req=350). proposed w2.0 vs baseline w2.0

### X = G2

| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |
|---|---|---|---|
| (7a) converged ∧ G2_post | 1.000 [0.989, 1.000] (350/350) | 0.211 [0.172, 0.257] (74/350) | 0.789 [0.742, 0.828] |
| (2'-G2) converged AND G2_post AND near the target (gamma after relaxation) | 0.260 [0.217, 0.308] (91/350) | 0.011 [0.004, 0.029] (4/350) | 0.249 [0.202, 0.297] |
| (4-G2) converged AND novel AND G2_post | 0.983 [0.963, 0.992] (344/350) | 0.211 [0.172, 0.257] (74/350) | 0.771 [0.722, 0.812] |
| (6-post-G2) novel, unique, near the target (connected components of the induced subgraph) | 0.257 [0.214, 0.305] (90/350) | 0.011 [0.004, 0.029] (4/350) | 0.246 [0.199, 0.294] |
| (1'-G2) median abs(gamma_relaxed-gamma_target) (converged AND G2_post only; count shown) | 0.405 (n=350) | 0.509 (n=74) | -0.104 |

### X = E

| metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |
|---|---|---|---|
| (7b) converged ∧ E_post | 1.000 [0.989, 1.000] (350/350) | 0.000 [0.000, 0.011] (0/350) | 1.000 [0.985, 1.000] |
| (2'-E) converged AND E_post AND near the target (gamma after relaxation) | 0.260 [0.217, 0.308] (91/350) | 0.000 [0.000, 0.011] (0/350) | 0.260 [0.215, 0.308] |
| (4-E) converged AND novel AND E_post | 0.983 [0.963, 0.992] (344/350) | 0.000 [0.000, 0.011] (0/350) | 0.983 [0.960, 0.992] |
| (6-post-E) novel, unique, near the target (connected components of the induced subgraph) | 0.257 [0.214, 0.305] (90/350) | 0.000 [0.000, 0.011] (0/350) | 0.257 [0.213, 0.305] |
| (1'-E) median abs(gamma_relaxed-gamma_target) (converged AND E_post only; count shown) | 0.405 (n=350) | NA (n=0) | NA |

## Reference (pre side; not used for claims)

| target set | X | metric | proposed | baseline | proposed - baseline [Newcombe 95% CI] |
|---|---|---|---|---|---|
| inner5 | G2 | (7a-pre) valid_pre ∧ G2_pre | 1.000 [0.985, 1.000] (250/250) | 0.248 [0.199, 0.305] (62/250) | 0.752 [0.693, 0.801] |
| inner5 | G2 | (2-G2) valid_pre AND G2_pre AND near the target (gamma before relaxation) | 0.388 [0.330, 0.450] (97/250) | 0.052 [0.031, 0.087] (13/250) | 0.336 [0.268, 0.401] |
| inner5 | G2 | (4-G2-pre) valid_pre AND novel AND G2_pre | 0.992 [0.971, 0.998] (248/250) | 0.248 [0.199, 0.305] (62/250) | 0.744 [0.683, 0.794] |
| inner5 | G2 | (6-pre-G2) | 0.384 [0.326, 0.446] (96/250) | 0.052 [0.031, 0.087] (13/250) | 0.332 [0.264, 0.397] |
| inner5 | E | (7b-pre) valid_pre ∧ E_pre | 1.000 [0.985, 1.000] (250/250) | 0.000 [0.000, 0.015] (0/250) | 1.000 [0.979, 1.000] |
| inner5 | E | (2-E) valid_pre AND E_pre AND near the target (gamma before relaxation) | 0.388 [0.330, 0.450] (97/250) | 0.000 [0.000, 0.015] (0/250) | 0.388 [0.328, 0.450] |
| inner5 | E | (4-E-pre) valid_pre AND novel AND E_pre | 0.992 [0.971, 0.998] (248/250) | 0.000 [0.000, 0.015] (0/250) | 0.992 [0.966, 0.998] |
| inner5 | E | (6-pre-E) | 0.384 [0.326, 0.446] (96/250) | 0.000 [0.000, 0.015] (0/250) | 0.384 [0.324, 0.446] |
| all7 | G2 | (7a-pre) valid_pre ∧ G2_pre | 1.000 [0.989, 1.000] (350/350) | 0.211 [0.172, 0.257] (74/350) | 0.789 [0.742, 0.828] |
| all7 | G2 | (2-G2) valid_pre AND G2_pre AND near the target (gamma before relaxation) | 0.277 [0.233, 0.326] (97/350) | 0.037 [0.022, 0.063] (13/350) | 0.240 [0.189, 0.291] |
| all7 | G2 | (4-G2-pre) valid_pre AND novel AND G2_pre | 0.994 [0.979, 0.998] (348/350) | 0.211 [0.172, 0.257] (74/350) | 0.783 [0.735, 0.823] |
| all7 | G2 | (6-pre-G2) | 0.274 [0.230, 0.323] (96/350) | 0.037 [0.022, 0.063] (13/350) | 0.237 [0.186, 0.288] |
| all7 | E | (7b-pre) valid_pre ∧ E_pre | 1.000 [0.989, 1.000] (350/350) | 0.000 [0.000, 0.011] (0/350) | 1.000 [0.985, 1.000] |
| all7 | E | (2-E) valid_pre AND E_pre AND near the target (gamma before relaxation) | 0.277 [0.233, 0.326] (97/350) | 0.000 [0.000, 0.011] (0/350) | 0.277 [0.232, 0.326] |
| all7 | E | (4-E-pre) valid_pre AND novel AND E_pre | 0.994 [0.979, 0.998] (348/350) | 0.000 [0.000, 0.011] (0/350) | 0.994 [0.976, 0.998] |
| all7 | E | (6-pre-E) | 0.274 [0.230, 0.323] (96/350) | 0.000 [0.000, 0.011] (0/350) | 0.274 [0.229, 0.323] |

## Stratified exclusions (why a structure is not G2. arm x target x stage. The denominator is 50 each.)

| arm | target | stage | G2 | no_gb | multi_gb | disordered | excluded_pre | other |
|---|---|---|---|---|---|---|---|---|
| proposed | 0 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 0 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 1 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 1 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 2 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 2 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 3 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 3 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 4 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 4 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 5 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 5 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 6 | post | 50 | 0 | 0 | 0 | 0 | 0 |
| proposed | 6 | pre | 50 | 0 | 0 | 0 | 0 | 0 |
| baseline | 0 | post | 3 | 43 | 0 | 0 | 4 | 0 |
| baseline | 0 | pre | 3 | 43 | 0 | 0 | 4 | 0 |
| baseline | 1 | post | 8 | 41 | 0 | 0 | 1 | 0 |
| baseline | 1 | pre | 8 | 41 | 0 | 0 | 1 | 0 |
| baseline | 2 | post | 11 | 39 | 0 | 0 | 0 | 0 |
| baseline | 2 | pre | 11 | 39 | 0 | 0 | 0 | 0 |
| baseline | 3 | post | 18 | 30 | 0 | 0 | 2 | 0 |
| baseline | 3 | pre | 18 | 30 | 0 | 0 | 2 | 0 |
| baseline | 4 | post | 22 | 27 | 0 | 0 | 1 | 0 |
| baseline | 4 | pre | 22 | 27 | 0 | 0 | 1 | 0 |
| baseline | 5 | post | 1 | 38 | 0 | 0 | 11 | 0 |
| baseline | 5 | pre | 1 | 38 | 0 | 0 | 11 | 0 |
| baseline | 6 | post | 11 | 38 | 0 | 0 | 1 | 0 |
| baseline | 6 | pre | 11 | 38 | 0 | 0 | 1 | 0 |

- Calls of UNI.fit (connected components): G2 version 14 groups, 2436 calls, 0 edges / E version 8 groups, 2403 calls, 0 edges.
- The existing metrics and outputs (`experiments/relax_novelty/results/relax_novelty_summary.json`) are unchanged. For the exact agreement of F2 see validation.json.


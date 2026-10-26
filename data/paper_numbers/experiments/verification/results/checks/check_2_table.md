# Machine check table v2 (check 2 of the DFT validation; after the replacement of the DFT inputs)

Status: complete. No interpretation or recommendation. The decisions are only PASS / FAIL / FLAG. The earlier table `check_2.md` was not overwritten.
Abbreviations: PLAN = the plan of the DFT validation, RV = the verification experiment directory, V2 = RV/results/dft_recalc, M = RV/dft_results_merged

## Possibility that my own checks do not fail (stated first)
- The recomputations of items 2, 4 and 5 take the values of V2/gamma_vasp.csv (E_toten, mu, N, A, gamma_mace) as inputs, as the implementation does. In item 6, however, E_toten of the csv (26 rows) and mu_calcs.e_toten of the verdict json (13) were compared directly with the final TOTEN of the OUTCAR files (all 39 agree within 1e-8 eV), so the origin of the csv values was confirmed down to the OUTCAR.
- The decision "cluster A / cluster B" of item 7 is based on the names of the link-target directories of the symbolic links in M (`../dft_results/` = cluster A, `../dft_results_b/` = cluster B). The correspondence "dft_results is the cluster-A part" follows the description in the request (the provenance of the link-target directories themselves was not examined). The VASP versions in the first line of the OUTCAR files (27 on cluster A = 6.4.3, 12 on cluster B = 6.4.2) were consistent with the submission notes.
- The convergence decision of item 6 uses regular expressions on the OUTCAR body (the last `Iteration N( M)` line and the presence of `aborting loop because EDIFF is reached`). NSW=0, so there is only one ionic step of electronic steps.

## Check table

| # | item | definition in the master (file:line) | measured value (file:line/column, my own recomputation) | decision | agreement with the implementation output |
|---|---|---|---|---|---|
| 1 | The selection is 13 pairs, 26 structures (proposed 7, baseline 6). Selection rule lo=floor(0.1(m-1)+0.5), hi=floor(0.9(m-1)+0.5); pairs with m<2 or lo==hi are excluded | PLAN:143 "13 pairs 26 structures (proposed 7, baseline 6), n_used=13. Stop if different", PLAN:104-105 | RV/results/dft_selection/selection.json: n_pairs=13, n_structures=26, n_used=13, arm counts of pairs proposed 7, baseline 6, excluded=[baseline target5 m=1 reason m<2]. proposed m=50 x7 (lo=5, hi=44), baseline m=[3,8,11,18,22,11] ((lo,hi)=(0,2),(1,6),(1,9),(2,15),(2,19),(1,9)). I computed the rule myself and all 13 agree, lo != hi. (chunk, cif_index), gamma_relaxed and natoms of selection.json agree with chunk, cif_index, gamma_mace and N of V2/gamma_vasp.csv for all 26 structures (0 mismatches) | PASS | agrees |
| 2 | Recomputation of gamma_DFT for all rows (E-N mu)/(2A) x 16.0218, tolerance relative 1e-9 | PLAN:115, the tolerance is from the request | Recomputed for all 26 rows of V2/gamma_vasp.csv (columns E_toten, N, mu, A, gamma_dft) (empty rows 0): maximum relative difference 0 (26/26 within 1e-9). The mu of the 14 proposed structures is the common mu (mu_name=mu_proposed_common, -3.7477348578333336); the 12 baseline structures have one mu_name per pair. The mu of the csv agrees with the mu of the json mu_calcs for all 13 within 1e-12, each mu = e_toten/120 (within 1e-12) | PASS | agrees with gamma_dft of the csv (26/26) |
| 3a | Existence of excluded pairs | PLAN:205 "excluded pairs are reported" | V2/verdict.json: n_unconverged=0, excluded_pairs=[], n_pairs=13. converged and ediff_reached of the csv are True for all 26 rows. No excluded pair | PASS (not applicable) | agrees |
| 3b | Pairs with abs(d gamma_MACE) < 0.04 | PLAN:115, PLAN:191 | abs(d gamma_MACE) of the 13 pairs computed by myself from gamma_mace of the csv: minimum 0.0813 (baseline_t1), maximum 0.7298 (proposed_t3). 0 pairs below 0.04. json n_excluded_small_delta=0 | PASS (not applicable) | agrees |
| 4 | Sign agreement and k | PLAN:115 "sign(gamma_DFT,hi - gamma_DFT,lo) == sign(gamma_MACE,hi - gamma_MACE,lo)" | All 13 pairs decided by myself: 13/13 signs agree, 0 disagree. n_used=13, k=13 (dMACE -> dDFT: proposed_t0 +0.3438 -> +0.4104, t1 +0.4049 -> +0.3697, t2 +0.4622 -> +0.3672, t3 +0.7298 -> +0.6252, t4 +0.4681 -> +0.3899, t5 +0.2579 -> +0.2714, t6 +0.6594 -> +0.5892, baseline_t0 +0.2435 -> +0.1419, t1 +0.0813 -> +0.0746, t2 +0.5355 -> +0.4563, t3 +0.3324 -> +0.3067, t4 +0.3229 -> +0.2788, t6 +0.3746 -> +0.3487) | PASS (k=13 / n_used=13) | agrees with k=13 and agree=true, status=used for the 13 pairs in the json |
| 5 | p and verdict | PLAN:191 PASS if p <= 0.01 / FLAG if 0.01 < p <= 0.05 or n_used < 7 / FAIL if p > 0.05. p = P(X >= k given n_used, 1/2) one-sided, exact with `fractions.Fraction` | My own computation (math.comb, Fraction): n_used=13, k=13 -> p=1/8192 (=0.0001220703125) -> p <= 0.01 and n_used >= 7 -> PASS. json function output: verdict=PASS, n_used=13, k=13, p="1/8192" | **PASS** | agrees with the implementation output (verdict, p, k, n_used all) |
| 6a | The OUTCAR of all 39 calculations converged (final `Iteration N(M)` with M < 200 and `aborting loop because EDIFF is reached` present) | the request. (PLAN:113 NELM=200) | I read the OUTCAR files of the 39 directories in M (excluding index.json) myself: 0 unconverged. The maximum M of the last `Iteration` is 33 (grain boundary) and 12 for all mu calculations. The EDIFF-reached text is present in 39/39 | PASS | agrees with n_calcs=39 and n_unconverged=0 of the json. n_elec_steps of the csv agrees with the last M of the OUTCAR for 26/26 |
| 6b | Agreement of E_toten of the csv with the final TOTEN of the OUTCAR | the request | Final `free  energy   TOTEN` of the OUTCAR vs E_toten of the csv: 26/26 agree within 1e-8 eV. json mu_calcs.e_toten: 13/13 agree. N of the csv vs NIONS of the OUTCAR: 26/26 agree; nions=120 of the mu calculations vs NIONS of the OUTCAR: 13/13 agree | PASS | agrees |
| 7 | Whether the 4 calculations of each pair (grain boundary hi and lo, mu hi and lo; proposed shares the common mu) were run in the same computing environment (decided by the link target) | the request | Link targets in M: cluster A (`../dft_results/`) 27, cluster B (`../dft_results_b/`) 12, UNKNOWN 0. Per pair: proposed_t0 to t6 = 2 grain boundaries + common mu all on cluster A (uniform). baseline_t1, t2, t3 = all 4 calculations on cluster A (uniform). baseline_t0, t4, t6 = all 4 calculations on cluster B (uniform). **Mixed pairs: 0**. The 12 OUTCAR files of cluster B are VASP 6.4.2 20Jul23, the 27 of cluster A are VASP 6.4.3 19Mar24 (line 1 of the OUTCAR). The scheduler log numbers in each cluster-B directory agree for all 12 with the valid job IDs of the submission notes, and the FINISHED times are 16:14 to 16:45 JST | PASS (no mixing). The VASP version differs between pairs (report only; not specified in the PLAN. See the note below) | agrees with the link targets |
| 8a | Earlier FLAG-1 (the handling of pairs with unfinished DFT was not preregistered) | FLAG-1 of the earlier table | This time there are 0 unfinished or unconverged pairs (3a, 6a), 0 exclusions, n_used=13, k=13. Whether a rule that defines the "handling of unfinished pairs" exists is the same as before because the PLAN was not changed (no description in PLAN:191). With the present input, however, there is no situation in which that rule applies | No situation of application with the present input (resolved = the event disappeared. The rule itself remains unregistered in the PLAN) | - |
| 8b | Earlier FLAG-2 (disagreement of n_used, 13 vs 10) | PLAN:115 "n_used=13 is fixed before running VASP" | json n_used=13, selection.json n_used=13, my own computation n_used=13. Agrees with PLAN:115 | Resolved (agrees) | - |
| 8c | Earlier FLAG-3 (interpretation of "after all DFT finished") | PLAN:191 column of the decision timing "after all DFT finished" | All 39 calculations have finished and converged (6a). Satisfied under both readings, "finished = terminated" and "finished = converged" | Resolved (satisfied under both readings) | - |

## Related reference values (not subject to decision; reported only. PLAN:116 "report only")
- V2/verdict.json report_only: Spearman rho all 0.9788, proposed 0.9956, baseline 0.9510. Median offset (DFT - MACE) proposed -0.1191, baseline -0.1769 J/m2. MAE after offset removal proposed 0.0571, baseline 0.0730 J/m2. These are transcribed from the implementation output and not recomputed by myself.
- The value of the present run that PLAN:115 asks to report together with the basis of 0.04 J/m2 (an earlier MAE of 0.0217) is "to be reported together when the value is available". The MAE above (0.0571, 0.0730) may apply, but the work of reporting it together is not done on the checking side.

## FLAG details
- No item with the decision FLAG.
- Note (facts only): the 12 calculations of 3 pairs (baseline_t0, t4, t6) use VASP 6.4.2 and the calculations of the other 10 pairs use VASP 6.4.3. PLAN:113 specifies INCAR, KPOINTS, POTCAR and the input structures but does not specify the VASP version or a uniform computer (I read PLAN:101-116 through; there is no description of the version). Because it is not specified, it is not a subject of the decision. The submission note states that the difference of E_TOTEN is "expected to be about EDIFF=1E-6 (unverified)" (it remains unverified). There is no comparison of the same input computed in two environments, so this effect was not measured.
- Note (facts only): the point of the earlier FLAG-1 that "the PLAN has no rule for unfinished pairs" remains because the PLAN is not revised, but it does not apply because the present input has no unfinished pairs.

## Overall result of this check (mechanical results only)
- Return value of the implemented decision function: PASS (n_used=13, k=13, p=1/8192). Agrees with my recomputation.
- Overall on the checking side: **PASS** (p=1/8192 <= 0.01, n_used=13 >= 7). No reason for FLAG.
- The validation of the decision function was confirmed earlier. This time I recomputed the p of the 4 cases of PLAN:213 myself and they agree (13,12 -> 7/4096 PASS region, 13,9 -> 1093/8192 FAIL region, 13,11 -> 23/2048 FLAG region, 6,6 -> 1/64 n<7). 14/8192 and 92/8192 of PLAN:213 are equal values before reduction.
- By PLAN:197 this check is not included in the composition of the overall decision (only the master description is quoted).

## Items that contain guesses
- The correspondence "`../dft_results/` = cluster A" of item 7 is based on the description in the request (the provenance of the link-target directories was not confirmed; it is consistent with the difference in the version text of the OUTCAR files).
- "The rule itself remains unregistered in the PLAN" in item 8a is a claim of absence based on the earlier grep result and on reading the line ranges of the PLAN through (the whole PLAN was not grepped again this time).
- Spearman and MAE in the report-only column are transcribed from the implementation output (not recomputed).

## Commands used
```
python3 gk2.py   # standard library only. csv/json/OUTCAR regular expressions, binomial with Fraction, link-target decision
```

## Files used
- PLAN:101-116,143-145,185-216 (read ranges 95-124,138-150,185-220)
- RV/results/dft_recalc/gamma_vasp.csv、RV/results/dft_recalc/verdict.json
- RV/results/dft/selection.json、RV/results/dft/dft_results_b_submit.md
- RV/dft_results_merged/<39 names>/OUTCAR (symbolic link targets), the scheduler logs and FINISHED of the 12 on cluster B
- Earlier: RV/results/checks/check_2.md (read only)
